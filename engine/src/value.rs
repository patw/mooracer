//! `Value` — the compact native value tree of a MooRacer document.
//!
//! Design notes (perf posture, see spec "Data model" / "Performance posture"):
//!
//! - One enum, no `serde_json::Value`. `Bool`/`I64`/`F64` are stored inline in
//!   the discriminant slot so scalar values are cheap to compare and hash-route.
//! - `Object` is an **insertion-ordered** `Vec<(String, Value)>`. Small docs
//!   (the common case) stay on one or two cache lines, iteration order matches
//!   document order (needed later for deterministic `first`/`last` aggregation
//!   and stable index rebuilds), and `get` is a linear key scan — O(1) for the
//!   typical 2–10 field document. If profiling later shows key lookups as a
//!   hotspot we add a per-object key index; a `BTreeMap` was rejected up front
//!   because it destroys insertion order and is allocation-heavy per key.
//! - No `unsafe` here: it would buy nothing at this level (enum + Vec), and
//!   the spec allows `unsafe` only where it is measured.
//!
//! Path syntax (used by [`Value::get_path`] / [`Value::set_path`]):
//!
//! - Segments are separated by `.`, or wrapped in `[...]` (both work:
//!   `a.b.c`, `a[0].b`, `a.b[2][3]`).
//! - On an `Object` node a token is **always a key** (so numeric-looking keys
//!   like `"0"` work).
//! - On an `Array` node a token must be an unsigned decimal **index**;
//!   otherwise the path does not resolve.
//!
//! `set_path` semantics (MongoDB-compatible for the cases the spec cares
//! about; recorded in spec.md "Value model"):
//!
//! - The leaf is inserted/replaced in an object, replaced in-range in an
//!   array, or appended when the index equals the array length.
//! - A missing intermediate *object* field is created as a new empty object.
//! - A missing intermediate *array* field whose next segment is an index is
//!   created as an array padded with `Null` up to that index (sparse create).
//! - Descending past an array that is too short (index > length) or into a
//!   scalar is a [`PathError`].

use std::cmp::{Ordering, PartialOrd};
use std::fmt;
use std::fmt::Write as _;

/// A MooRacer document value: JSON-like tree, native Rust layout.
// NOTE: `PartialEq`/`Eq` are implemented manually (cross-numeric +
// canonical-object semantics) — do not derive them.
#[derive(Clone, Debug, Default)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
    Array(Vec<Value>),
    /// Insertion-ordered key/value pairs. Keys are unique (all mutators keep
    /// that invariant; [`Value::get`] finds the first match, which is the
    /// only match).
    Object(Vec<(String, Value)>),
}

/// Errors from path operations on [`Value`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathError {
    /// The path was empty or structurally invalid (e.g. `a..b`, `a[0]b`,
    /// trailing `[`).
    InvalidPath(String),
    /// A token that must be an array index was not a non-negative integer
    /// (only possible on arrays — object tokens are always keys).
    NotAnIndex(String),
    /// Array index out of range for an existing array (`index > len`).
    IndexOutOfRange { index: usize, len: usize },
    /// Tried to descend into a scalar (path has segments past a leaf).
    CannotDescend { found: &'static str },
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathError::InvalidPath(p) => write!(f, "invalid path: {p:?}"),
            PathError::NotAnIndex(t) => write!(f, "{t:?} is not an array index"),
            PathError::IndexOutOfRange { index, len } => {
                write!(f, "array index {index} out of range (len {len})")
            }
            PathError::CannotDescend { found } => write!(f, "cannot descend into {found}"),
        }
    }
}

impl std::error::Error for PathError {}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl Value {
    /// `Null` value.
    pub const fn null() -> Self {
        Value::Null
    }
    /// Boolean value.
    pub const fn bool(b: bool) -> Self {
        Value::Bool(b)
    }
    /// Signed 64-bit integer value.
    pub const fn i64(n: i64) -> Self {
        Value::I64(n)
    }
    /// 64-bit float value.
    pub const fn f64(x: f64) -> Self {
        Value::F64(x)
    }
    /// String value (allocates).
    pub fn str(s: impl Into<String>) -> Self {
        Value::Str(s.into())
    }
    /// Empty array value.
    pub const fn array() -> Self {
        Value::Array(Vec::new())
    }
    /// Array value from a pre-built `Vec` (no copy).
    pub fn array_from(items: Vec<Value>) -> Self {
        Value::Array(items)
    }
    /// Empty object value.
    pub const fn object() -> Self {
        Value::Object(Vec::new())
    }
    /// Object value from pre-built ordered pairs (no copy). Pairs must have
    /// unique keys (all public mutators keep that invariant).
    pub fn object_from(pairs: Vec<(String, Value)>) -> Self {
        Value::Object(pairs)
    }
    /// Integer that does not fit in `i64` (e.g. a large `u64`): stored as
    /// `F64`, best-effort (may round above 2^53).
    pub fn from_u64(n: u64) -> Self {
        if n <= i64::MAX as u64 {
            Value::I64(n as i64)
        } else {
            Value::F64(n as f64)
        }
    }
}

// ---------------------------------------------------------------------------
// Type predicates / introspection
// ---------------------------------------------------------------------------

impl Value {
    pub const fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
    pub const fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }
    pub const fn is_i64(&self) -> bool {
        matches!(self, Value::I64(_))
    }
    pub const fn is_f64(&self) -> bool {
        matches!(self, Value::F64(_))
    }
    /// True for `I64` or `F64`.
    pub const fn is_number(&self) -> bool {
        matches!(self, Value::I64(_) | Value::F64(_))
    }
    pub const fn is_str(&self) -> bool {
        matches!(self, Value::Str(_))
    }
    pub const fn is_array(&self) -> bool {
        matches!(self, Value::Array(_))
    }
    pub const fn is_object(&self) -> bool {
        matches!(self, Value::Object(_))
    }
    /// True for `Null | Bool | I64 | F64 | Str`.
    pub const fn is_scalar(&self) -> bool {
        matches!(self, Value::Null | Value::Bool(_) | Value::I64(_) | Value::F64(_) | Value::Str(_))
    }
    /// `true` for `Array` and `Object`.
    pub const fn is_container(&self) -> bool {
        matches!(self, Value::Array(_) | Value::Object(_))
    }
    /// Stable type name: `null`, `bool`, `i64`, `f64`, `str`, `array`,
    /// `object`.
    pub const fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::I64(_) => "i64",
            Value::F64(_) => "f64",
            Value::Str(_) => "str",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
    /// Element count for `Array`/`Object`; `0` for scalars.
    pub fn len(&self) -> usize {
        match self {
            Value::Array(v) => v.len(),
            Value::Object(v) => v.len(),
            _ => 0,
        }
    }
    /// `true` for empty containers and all scalars.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl Value {
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// Strict: `I64` only. (Numeric *comparison* is cross-type — see `Ord`.)
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I64(n) => Some(*n),
            _ => None,
        }
    }
    /// Cross-type numeric accessor: `I64` is widened, `F64` passed through.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::I64(n) => Some(*n as f64),
            Value::F64(x) => Some(*x),
            _ => None,
        }
    }
    /// Checked conversion from an `I64` to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::I64(n) => u64::try_from(*n).ok(),
            _ => None,
        }
    }
    /// Borrowed string content.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_object(&self) -> Option<&Vec<(String, Value)>> {
        match self {
            Value::Object(v) => Some(v),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Object mutators / helpers
// ---------------------------------------------------------------------------

impl Value {
    /// Top-level key lookup on an object. `None` on a miss or a non-object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries
                .iter()
                .find(|(k, _)| k.as_str() == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
    /// Ordered key iterator for an object (empty for non-objects).
    pub fn keys(&self) -> impl Iterator<Item = &str> + '_ {
        let entries: &[(String, Value)] = match self {
            Value::Object(e) => e,
            _ => &[],
        };
        entries.iter().map(|(k, _)| k.as_str())
    }
    /// Ordered `(key, value)` iterator for an object.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> + '_ {
        let entries: &[(String, Value)] = match self {
            Value::Object(e) => e,
            _ => &[],
        };
        entries.iter().map(|(k, v)| (k.as_str(), v))
    }
    /// Insert or replace `key` in an object (no-op on a non-object).
    /// Returns the previous value, if any.
    pub fn set(&mut self, key: &str, value: Value) -> Option<Value> {
        match self {
            Value::Object(entries) => {
                if let Some((_, v)) = entries.iter_mut().find(|(k, _)| k.as_str() == key) {
                    Some(std::mem::replace(v, value))
                } else {
                    entries.push((key.to_string(), value));
                    None
                }
            }
            _ => None,
        }
    }
    /// Remove `key` from an object. `None` on a miss or a non-object.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        match self {
            Value::Object(entries) => {
                let pos = entries.iter().position(|(k, _)| k.as_str() == key)?;
                Some(entries.remove(pos).1)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Path-based access
// ---------------------------------------------------------------------------

/// Split `path` into borrowed tokens, validating structure. No allocation:
/// tokens are slices of `path`.
///
/// Rules: segments separated by `.` or `[` `]`. Interpretation (index vs
/// key) is deferred to the node type — an object token is always a key, an
/// array token must parse as an unsigned decimal index — so `{"0": …}` keys
/// keep working with dotted or bracketed paths alike.
fn parse_path(path: &str) -> Result<Vec<&str>, PathError> {
    let b = path.as_bytes();
    let mut toks = Vec::new();
    let mut start = 0usize;
    let mut in_bracket = false;
    let mut had_bracket = false;
    for (i, c) in b.iter().enumerate() {
        match c {
            b'.' if !in_bracket => {
                let s = &path[start..i];
                if s.is_empty() {
                    if !had_bracket {
                        // `..`, leading `.`, or empty path.
                        return Err(PathError::InvalidPath(path.to_string()));
                    }
                    // `]` immediately followed by `.`: the token was already
                    // pushed at `]` (e.g. `a[0].b`).
                } else if had_bracket {
                    // bare chars between `]` and `.` are junk (`a[0]b.c`).
                    return Err(PathError::InvalidPath(path.to_string()));
                } else {
                    toks.push(s);
                }
                start = i + 1;
                had_bracket = false;
            }
            b'[' if !in_bracket => {
                // `a.b[` form: flush the bare token accumulated so far, then
                // start a bracketed token.
                let s = &path[start..i];
                if s.is_empty() {
                    if !had_bracket {
                        // `[` with nothing before it: either the start of the
                        // path (a path must start with a bare key) or junk
                        // right after a `.`.
                        return Err(PathError::InvalidPath(path.to_string()));
                    }
                    // chained brackets (`a[0][1]`): nothing to flush.
                } else if had_bracket {
                    // bare chars between `]` and `[` are junk (`a[0]x[1]`).
                    return Err(PathError::InvalidPath(path.to_string()));
                } else {
                    toks.push(s);
                }
                in_bracket = true;
                start = i + 1;
                had_bracket = false;
            }
            b']' if in_bracket => {
                let s = &path[start..i];
                if s.is_empty() {
                    return Err(PathError::InvalidPath(path.to_string()));
                }
                toks.push(s);
                in_bracket = false;
                had_bracket = true;
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = &path[start..];
    if in_bracket {
        return Err(PathError::InvalidPath(path.to_string())); // dangling `[`
    }
    if had_bracket {
        // Path ended with a closed bracket (`a[0]`, `a.b[2][3]`): the last
        // token was already pushed at `]`. Anything after it is junk.
        if !tail.is_empty() {
            return Err(PathError::InvalidPath(path.to_string()));
        }
        return Ok(toks);
    }
    // Bare trailing token: must be non-empty ("", trailing `.` and a leading
    // empty key are rejected here or at the `.` boundary).
    if tail.is_empty() {
        return Err(PathError::InvalidPath(path.to_string()));
    }
    toks.push(tail);
    Ok(toks)
}

/// Interpret a token against an array: non-negative integer index.
fn tok_index(tok: &str) -> Option<usize> {
    // Tokens are `[0-9]+` only if they parsed as an index; reject signs,
    // whitespace, exponent forms.
    if tok.is_empty() || !tok.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    tok.parse::<usize>().ok()
}

/// True when a token *could* be an index (all digits) — used to decide
/// whether a missing intermediate field becomes a padded array.
fn tok_is_index_like(tok: &str) -> bool {
    !tok.is_empty() && tok.bytes().all(|c| c.is_ascii_digit())
}

/// The container to create for a missing/null intermediate field, decided by
/// looking one token ahead: an index-like next token means a padded sparse
/// array (MongoDB-style), otherwise an empty object.
fn container_for(rest: &[&str]) -> Value {
    match rest.first() {
        Some(next) if tok_is_index_like(next) => {
            let idx = next.parse::<usize>().unwrap_or(0);
            Value::Array(vec![Value::Null; idx + 1])
        }
        _ => Value::Object(Vec::new()),
    }
}

/// Ensure `node` is a container that can be descended into for `rest`:
/// convert a `Null` slot in place (Mongo converts nulls), recurse into an
/// existing container, or fail on a non-null scalar.
fn ensure_container(node: &mut Value, rest: &[&str]) -> Result<(), PathError> {
    match node {
        Value::Null => {
            *node = container_for(rest);
            Ok(())
        }
        Value::Array(_) | Value::Object(_) => Ok(()),
        other => Err(PathError::CannotDescend { found: other.type_name() }),
    }
}

impl Value {
    /// Resolve a path to a borrowed value. Any non-matching step (missing
    /// key, non-index token on an array, index past the end, scalar mid
    /// path) yields `None` — there is no error channel for reads.
    pub fn get_path(&self, path: &str) -> Option<&Value> {
        let toks = parse_path(path).ok()?;
        let mut node = self;
        for tok in &toks {
            node = match node {
                Value::Object(entries) => {
                    let (_, v) = entries.iter().find(|(k, _)| k.as_str() == *tok)?;
                    v
                }
                Value::Array(items) => {
                    let i = tok_index(tok)?;
                    items.get(i)?
                }
                _ => return None,
            };
        }
        Some(node)
    }

    /// Set the value at `path`, creating missing intermediate objects (and
    /// sparse arrays for missing intermediate array fields, see module
    /// docs). See the module-level "set_path semantics" for the exact rules.
    pub fn set_path(&mut self, path: &str, value: Value) -> Result<(), PathError> {
        let toks = parse_path(path)?;
        set_path_inner(self, &toks, value)
    }

    /// Remove the value at `path` (the inverse of [`Value::set_path`]).
    ///
    /// - On an `Object` the key at `path` is removed (including the `_id`
    ///   rule-agnostic: this is a raw tree op, the collection layer keeps the
    ///   invariant that `_id` is never unset);
    /// - On an `Array` the element at the index is removed, shifting later
    ///   elements down; an index `== len` (nothing to remove) or past the
    ///   end is an error, and a non-index token on an array is an error.
    ///
    /// A missing intermediate step (key or index that does not exist) is
    /// **not** an error: nothing is removed and `false` is returned. Only a
    /// structurally invalid path, an out-of-range array index at the leaf,
    /// or descending into a scalar is a [`PathError`].
    pub fn remove_path(&mut self, path: &str) -> Result<bool, PathError> {
        let toks = parse_path(path)?;
        remove_path_inner(self, &toks)
    }
}

/// Recursive path-remove. `toks` is non-empty (guaranteed by `parse_path`):
/// `head` is the next segment, `rest` the remainder (empty at the leaf step).
fn remove_path_inner(node: &mut Value, toks: &[&str]) -> Result<bool, PathError> {
    let (head, rest) = toks.split_first().expect("parse_path yields >= 1 token");
    match node {
        Value::Object(entries) => match entries.iter().position(|(k, _)| k.as_str() == *head) {
            Some(pos) if rest.is_empty() => {
                entries.remove(pos);
                Ok(true)
            }
            Some(pos) => remove_path_inner(&mut entries[pos].1, rest),
            None => Ok(false), // key missing at this level: nothing removed
        },
        Value::Array(items) => {
            let idx = tok_index(head).ok_or_else(|| PathError::NotAnIndex(head.to_string()))?;
            if rest.is_empty() {
                match idx.cmp(&items.len()) {
                    Ordering::Less => {
                        items.remove(idx);
                        Ok(true)
                    }
                    Ordering::Greater => Err(PathError::IndexOutOfRange {
                        index: idx,
                        len: items.len(),
                    }),
                    // idx == len: no element there to remove (no-op)
                    Ordering::Equal => Ok(false),
                }
            } else {
                if idx >= items.len() {
                    return Ok(false); // missing intermediate: nothing removed
                }
                remove_path_inner(&mut items[idx], rest)
            }
        }
        _ => Err(PathError::CannotDescend { found: node.type_name() }),
    }
}

/// Recursive path-set. `toks` is guaranteed non-empty by `parse_path`:
/// `head` is the next segment to consume, `rest` the remainder (empty at
/// the leaf step, which writes `value` directly).
fn set_path_inner(node: &mut Value, toks: &[&str], value: Value) -> Result<(), PathError> {
    let (head, rest) = toks.split_first().expect("parse_path yields >= 1 token");
    match node {
        Value::Object(entries) => match entries.iter_mut().find(|(k, _)| k.as_str() == *head) {
            Some((_, v)) => {
                if rest.is_empty() {
                    *v = value;
                    Ok(())
                } else {
                    ensure_container(v, rest)?;
                    set_path_inner(v, rest, value)
                }
            }
            None => {
                if rest.is_empty() {
                    entries.push((head.to_string(), value));
                    Ok(())
                } else {
                    let created = container_for(rest);
                    entries.push((head.to_string(), created));
                    set_path_inner(&mut entries.last_mut().unwrap().1, rest, value)
                }
            }
        },
        Value::Array(items) => {
            let idx = tok_index(head).ok_or_else(|| PathError::NotAnIndex(head.to_string()))?;
            if rest.is_empty() {
                match idx.cmp(&items.len()) {
                    Ordering::Less => {
                        items[idx] = value;
                        Ok(())
                    }
                    Ordering::Equal => {
                        items.push(value);
                        Ok(())
                    }
                    Ordering::Greater => Err(PathError::IndexOutOfRange {
                        index: idx,
                        len: items.len(),
                    }),
                }
            } else {
                // Intermediate: the slot must exist (or be created at `len`).
                if idx > items.len() {
                    return Err(PathError::IndexOutOfRange {
                        index: idx,
                        len: items.len(),
                    });
                }
                if idx == items.len() {
                    items.push(Value::Null);
                }
                let slot = &mut items[idx];
                ensure_container(slot, rest)?;
                set_path_inner(slot, rest, value)
            }
        }
        _ => Err(PathError::CannotDescend { found: node.type_name() }),
    }
}

// ---------------------------------------------------------------------------
// Ordering / equality (total, Mongo-numeric-compatible)
// ---------------------------------------------------------------------------
//
// Documented total order (this is what field indexes and `.sort()` use):
//
//   Null < Bool < Number (I64 | F64, exact cross-numeric) < Str < Array < Object
//
// - `I64` and `F64` share one band and compare **exactly** (dyadic integer
//   comparison, no precision loss at 2^53), so `1` and `1.0` are equal and
//   order correctly.
// - `NaN` compares equal to itself and orders *after* `+inf` (total order
//   requires a NaN position; NaNs are meaningless in indexed fields anyway).
// - `Object` compares **canonical** (keys in byte order), so key order in the
//   document does not affect equality or ordering — matching Mongo document
//   equality. `Array` compares element-wise.

const RANK_NULL: i8 = 0;
const RANK_BOOL: i8 = 1;
const RANK_NUMBER: i8 = 2;
const RANK_STR: i8 = 3;
const RANK_ARRAY: i8 = 4;
const RANK_OBJECT: i8 = 5;

fn rank(v: &Value) -> i8 {
    match v {
        Value::Null => RANK_NULL,
        Value::Bool(_) => RANK_BOOL,
        Value::I64(_) | Value::F64(_) => RANK_NUMBER,
        Value::Str(_) => RANK_STR,
        Value::Array(_) => RANK_ARRAY,
        Value::Object(_) => RANK_OBJECT,
    }
}

/// Exact ordering of an `i64` against a *finite* `f64`, without precision
/// loss. `f = mant * 2^exp` with a signed 53-bit `mant`; compare
/// `i * 2^(-exp)` against `mant` in `i128` (the shifts never overflow:
/// when they would, the sign of `i` alone decides).
fn cmp_i64_f64(i: i64, f: f64) -> Ordering {
    debug_assert!(f.is_finite());
    let bits = f.to_bits();
    if bits & 0x7FFF_FFFF_FFFF_FFFF == 0 {
        // ±0.0: the "implicit bit" does not apply to zero.
        return if i == 0 {
            Ordering::Equal
        } else {
            i.cmp(&0)
        };
    }
    let exp = ((bits >> 52) & 0x7FF) as i32 - 1075; // 1023 bias + 52 implicit bits
    // NOTE: the mantissa mask is 52 bits (0x000F_FFFF_FFFF_FFFF) and the
    // implicit leading bit is 1 << 52 (0x0010_0000_0000_0000) — both once
    // truncated to 48/36 bits silently corrupted every comparison.
    let mant = if bits >> 63 != 0 {
        -(((bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000) as i128)
    } else {
        ((bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000) as i128
    };
    // f = mant * 2^exp with |mant| in [2^52, 2^53). Compare in i128; when the
    // required scale overflows i128, the magnitude gap is so large that the
    // signs alone decide.
    if exp < 0 {
        // lhs = i * 2^(-exp), rhs = mant.
        let shift = (-exp) as u32; // 1..=1074
        if shift <= 63 {
            ((i as i128) << shift).cmp(&mant) // |i| < 2^63, shift <= 63: fits
        } else if i == 0 {
            0_i128.cmp(&mant)
        } else {
            // |i * 2^shift| >= 2^64 > |mant|: the sign of i decides.
            i.cmp(&0)
        }
    } else {
        // lhs = i, rhs = mant * 2^exp.
        if exp <= 74 {
            (i as i128).cmp(&(mant << exp)) // |mant| < 2^53, exp <= 74: fits
        } else if i == 0 {
            0_i128.cmp(&mant)
        } else {
            // |f| >= 2^127 > |i|: same sign => f has larger magnitude;
            // opposite sign => i's sign is the answer.
            if (i > 0) == (mant > 0) {
                i.cmp(&0).reverse()
            } else {
                i.cmp(&0)
            }
        }
    }
}

/// Total order on f64 for index/sort use: NaN == NaN, NaN > +inf.
fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // unreachable: partial_cmp only diverges on NaN; keep the total order
        (false, false) => Ordering::Equal,
    })
}

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::I64(a), Value::I64(b)) => a == b,
            (Value::I64(a), Value::F64(b)) => cmp_i64_f64(*a, *b) == Ordering::Equal,
            (Value::F64(a), Value::I64(b)) => cmp_i64_f64(*b, *a) == Ordering::Equal,
            (Value::F64(a), Value::F64(b)) => cmp_f64(*a, *b) == Ordering::Equal,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Object(a), Value::Object(b)) => cmp_objects_canonical(a, b) == Ordering::Equal,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Value) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Value) -> Ordering {
        match (self, other) {
            // Cross-numeric first: exact, precision-safe.
            (Value::I64(a), Value::F64(b)) => cmp_i64_f64(*a, *b),
            (Value::F64(a), Value::I64(b)) => cmp_i64_f64(*b, *a).reverse(),
            _ => {
                let c = rank(self).cmp(&rank(other));
                if c != Ordering::Equal {
                    return c;
                }
                match (self, other) {
                    (Value::Null, Value::Null) => Ordering::Equal,
                    (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
                    (Value::I64(a), Value::I64(b)) => a.cmp(b),
                    (Value::F64(a), Value::F64(b)) => cmp_f64(*a, *b),
                    (Value::Str(a), Value::Str(b)) => a.cmp(b),
                    (Value::Array(a), Value::Array(b)) => a.cmp(b),
                    (Value::Object(a), Value::Object(b)) => cmp_objects_canonical(a, b),
                    _ => Ordering::Equal,
                }
            }
        }
    }
}

/// Canonical (keys in byte order) comparison of two insertion-ordered
/// object entries lists — lexicographic over the sorted `(key, value)` pair
/// sequences: the first differing pair decides (keys first, then values),
/// and a true prefix sorts first. Consistent with `PartialEq` (equal key
/// sets + equal values). Each list is sorted by key first (a `Vec` of
/// references — one small allocation per object comparison; object
/// comparisons are rare on the hot path).
fn cmp_objects_canonical(a: &[(String, Value)], b: &[(String, Value)]) -> Ordering {
    let mut sa: Vec<&(String, Value)> = a.iter().collect();
    sa.sort_by(|x, y| x.0.cmp(&y.0));
    let mut sb: Vec<&(String, Value)> = b.iter().collect();
    sb.sort_by(|x, y| x.0.cmp(&y.0));
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < sa.len() && ib < sb.len() {
        match sa[ia].0.cmp(&sb[ib].0) {
            Ordering::Less => return Ordering::Less,
            Ordering::Greater => return Ordering::Greater,
            Ordering::Equal => {
                match sa[ia].1.cmp(&sb[ib].1) {
                    Ordering::Equal => {}
                    o => return o,
                }
                ia += 1;
                ib += 1;
            }
        }
    }
    if ia < sa.len() {
        Ordering::Greater // a has remaining pairs: a's sequence is longer
    } else if ib < sb.len() {
        Ordering::Less // b has remaining pairs
    } else {
        Ordering::Equal
    }
}

// ---------------------------------------------------------------------------
// From conversions (serde-free builders)
// ---------------------------------------------------------------------------

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
macro_rules! from_int {
    ($($t:ty),* $(,)?) => {$(
        impl From<$t> for Value {
            fn from(n: $t) -> Self {
                Value::I64(n as i64)
            }
        }
    )*};
}
from_int!(i8, i16, i32, i64, u8, u16, u32);

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::from_u64(n)
    }
}
impl From<f32> for Value {
    fn from(x: f32) -> Self {
        Value::F64(x as f64)
    }
}
impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Value::F64(x)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Str(s)
    }
}
impl From<&Value> for Value {
    fn from(v: &Value) -> Self {
        v.clone()
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::Array(v)
    }
}

/// Build an object from an iterator of `(key, value)` pairs (insertion
/// order preserved).
impl FromIterator<(String, Value)> for Value {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(it: T) -> Self {
        Value::Object(it.into_iter().collect())
    }
}

// ---------------------------------------------------------------------------
// Display — JSON-like rendering
// ---------------------------------------------------------------------------
//
// Human-facing / logging output, **not** a canonical codec (the wire format
// is FlatBuffers, landed in the network subtasks):
// - `F64` always renders with a decimal point (`1.0`, not `1`) so number
//   kinds stay distinguishable.
// - Non-finite floats render as `null` (JSON has no inf/nan).
// - Strings are JSON-escaped.

fn escape_json(s: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for c in s.chars() {
        match c {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            '\0' => f.write_str("\\u0000")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => f.write_char(c)?,
        }
    }
    Ok(())
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("null"),
            Value::Bool(true) => f.write_str("true"),
            Value::Bool(false) => f.write_str("false"),
            Value::I64(n) => write!(f, "{n}"),
            Value::F64(x) => {
                if x.is_finite() {
                    let s = x.to_string();
                    if s.contains('.') || s.contains('e') || s.contains('E') {
                        f.write_str(&s)
                    } else {
                        write!(f, "{s}.0")
                    }
                } else {
                    f.write_str("null")
                }
            }
            Value::Str(s) => {
                f.write_str("\"")?;
                escape_json(s, f)?;
                f.write_str("\"")
            }
            Value::Array(items) => {
                f.write_str("[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
            Value::Object(entries) => {
                f.write_str("{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str("\"")?;
                    escape_json(k, f)?;
                    f.write_str("\": ")?;
                    write!(f, "{v}")?;
                }
                f.write_str("}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    fn obj(pairs: &[(&str, Value)]) -> Value {
        Value::Object(pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
    }
    fn arr(items: &[Value]) -> Value {
        Value::Array(items.to_vec())
    }

    // -- variants & predicates ----------------------------------------------

    #[test]
    fn variants_and_predicates() {
        let null = Value::null();
        assert!(null.is_null() && !null.is_number() && null.is_scalar() && !null.is_container());
        assert_eq!(null.type_name(), "null");

        let b = Value::bool(true);
        assert!(b.is_bool() && b.is_scalar() && !b.is_number());
        assert_eq!(b.type_name(), "bool");

        let i = Value::i64(-42);
        assert!(i.is_i64() && i.is_number() && !i.is_f64());
        assert_eq!(i.type_name(), "i64");

        let f = Value::f64(1.5);
        assert!(f.is_f64() && f.is_number() && !f.is_i64());
        assert_eq!(f.type_name(), "f64");

        let s = Value::str("hi");
        assert!(s.is_str() && s.is_scalar());
        assert_eq!(s.type_name(), "str");

        let a = Value::array();
        assert!(a.is_array() && a.is_container() && a.is_empty() && a.len() == 0);
        assert_eq!(a.type_name(), "array");

        let o = Value::object();
        assert!(o.is_object() && o.is_container() && o.is_empty());
        assert_eq!(o.type_name(), "object");
    }

    #[test]
    fn constructors_and_len() {
        let a = Value::array_from(vec![Value::i64(1), Value::i64(2)]);
        assert_eq!(a.len(), 2);
        assert!(!a.is_empty());

        let o = Value::object_from(vec![("k".to_string(), Value::i64(1))]);
        assert_eq!(o.len(), 1);

        assert_eq!(Value::from_u64(1u64), Value::i64(1));
        assert_eq!(Value::from_u64(i64::MAX as u64), Value::i64(i64::MAX));
        assert_eq!(Value::from_u64(i64::MAX as u64 + 1), Value::f64((i64::MAX as u64 + 1) as f64));
        assert_eq!(Value::default(), Value::Null);
    }

    // -- From conversions ----------------------------------------------------

    #[test]
    fn from_impls() {
        assert_eq!(Value::from(true), Value::Bool(true));
        assert_eq!(Value::from(-7i8), Value::I64(-7));
        assert_eq!(Value::from(7i16), Value::I64(7));
        assert_eq!(Value::from(7i32), Value::I64(7));
        assert_eq!(Value::from(7i64), Value::I64(7));
        assert_eq!(Value::from(7u8), Value::I64(7));
        assert_eq!(Value::from(7u16), Value::I64(7));
        assert_eq!(Value::from(7u32), Value::I64(7));
        assert_eq!(Value::from(7u64), Value::I64(7));
        assert_eq!(Value::from(0.5f32), Value::F64(0.5));
        assert_eq!(Value::from(0.5f64), Value::F64(0.5));
        assert_eq!(Value::from("s"), Value::Str("s".to_string()));
        assert_eq!(Value::from(String::from("s")), Value::Str("s".to_string()));
        assert_eq!(Value::from(vec![Value::i64(1)]), arr(&[Value::i64(1)]));

        let mut v = Value::object();
        let v2: Value = std::iter::once(("k".to_string(), Value::i64(1))).collect();
        v.set("k", Value::i64(1));
        assert_eq!(v, v2);
    }

    // -- accessors -----------------------------------------------------------

    #[test]
    fn accessors() {
        assert_eq!(Value::i64(-1).as_i64(), Some(-1));
        assert_eq!(Value::f64(-1.0).as_i64(), None); // strict
        assert_eq!(Value::i64(-1).as_f64(), Some(-1.0)); // cross-type
        assert_eq!(Value::f64(-1.5).as_f64(), Some(-1.5));
        assert_eq!(Value::bool(true).as_f64(), None);
        assert_eq!(Value::i64(5).as_u64(), Some(5));
        assert_eq!(Value::i64(-5).as_u64(), None);
        assert_eq!(Value::f64(5.0).as_u64(), None);
        assert_eq!(Value::bool(true).as_bool(), Some(true));
        assert_eq!(Value::i64(1).as_bool(), None);
        assert_eq!(Value::str("x").as_str(), Some("x"));
        assert_eq!(Value::i64(1).as_str(), None);
        let a = arr(&[Value::i64(1)]);
        assert_eq!(a.as_array(), Some(&vec![Value::i64(1)]));
        assert!(Value::i64(1).as_array().is_none());
        let o = obj(&[("k", Value::i64(1))]);
        assert!(o.as_object().is_some());
        assert!(Value::array().as_object().is_none());
    }

    // -- object helpers --------------------------------------------------------

    #[test]
    fn object_helpers() {
        let mut o = Value::object();
        assert_eq!(o.get("a"), None);
        assert!(!o.contains_key("a"));

        o.set("a", Value::i64(1));
        o.set("b", Value::str("x"));
        assert_eq!(o.get("a"), Some(&Value::i64(1)));
        assert!(o.contains_key("b"));

        // replace keeps position and returns the old value
        assert_eq!(o.set("a", Value::i64(2)), Some(Value::i64(1)));
        assert_eq!(o.get("a"), Some(&Value::i64(2)));

        // insertion order preserved
        assert_eq!(o.keys().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(
            o.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>(),
            vec![("a".to_string(), Value::i64(2)), ("b".to_string(), Value::str("x"))]
        );

        assert_eq!(o.remove("b"), Some(Value::str("x")));
        assert!(!o.contains_key("b"));
        assert_eq!(o.remove("missing"), None);
        assert_eq!(o.get("a"), Some(&Value::i64(2)));

        // non-object: all no-ops / None
        let mut n = Value::i64(1);
        assert_eq!(n.get("a"), None);
        assert_eq!(n.set("a", Value::i64(1)), None);
        assert_eq!(n.remove("a"), None);
        assert_eq!(n.len(), 0);
    }

    // -- get_path --------------------------------------------------------------

    #[test]
    fn get_path_nested() {
        let doc = obj(&[
            ("a", obj(&[("b", obj(&[("c", Value::i64(7))]))])),
            ("arr", arr(&[obj(&[("x", Value::i64(1))]), obj(&[("x", Value::i64(2))])])),
        ]);
        assert_eq!(doc.get_path("a.b.c"), Some(&Value::i64(7)));
        assert_eq!(doc.get_path("a.b"), Some(&obj(&[("c", Value::i64(7))])));
        assert_eq!(doc.get_path("arr.0.x"), Some(&Value::i64(1)));
        assert_eq!(doc.get_path("arr.1.x"), Some(&Value::i64(2)));
        // bracket syntax
        assert_eq!(doc.get_path("arr[0].x"), Some(&Value::i64(1)));
        // bracket tokens on an object are keys: a[b] = {c: 7}, [c] -> 7
        assert_eq!(doc.get_path("a[b][c]"), Some(&Value::i64(7)));
        assert_eq!(doc.get_path("a.b[c]"), Some(&Value::i64(7)));
        // missing at every level
        assert_eq!(doc.get_path("a.b.d"), None);
        assert_eq!(doc.get_path("a.q.c"), None);
        assert_eq!(doc.get_path("q.c"), None);
        assert_eq!(doc.get_path("arr.2.x"), None);
        assert_eq!(doc.get_path("arr.99"), None);
        // scalar mid-path
        assert_eq!(doc.get_path("a.b.c.x"), None);
    }

    #[test]
    fn get_path_numeric_tokens_on_objects_are_keys() {
        // `{"0": {"1": 42}}` — dotted numeric paths hit keys, not indexes.
        let doc = obj(&[("0", obj(&[("1", Value::i64(42))]))]);
        assert_eq!(doc.get_path("0.1"), Some(&Value::i64(42)));
        assert_eq!(doc.get_path("0[1]"), Some(&Value::i64(42))); // bracket = key on object
        assert_eq!(doc.get_path("0.2"), None);
    }

    #[test]
    fn get_path_non_index_on_array_is_none() {
        let doc = arr(&[Value::i64(1), Value::i64(2)]);
        assert_eq!(doc.get_path("0"), Some(&Value::i64(1)));
        assert_eq!(doc.get_path("x"), None);
        assert_eq!(doc.get_path("-1"), None);
        assert_eq!(doc.get_path("1.0"), None); // 2 is a scalar: cannot descend
    }

    #[test]
    fn get_path_invalid_syntax_is_none() {
        let doc = obj(&[("a", Value::i64(1))]);
        assert_eq!(doc.get_path(""), None);
        assert_eq!(doc.get_path("a..b"), None);
        assert_eq!(doc.get_path("a."), None);
        assert_eq!(doc.get_path(".a"), None);
        assert_eq!(doc.get_path("a[0]b"), None);
        assert_eq!(doc.get_path("a["), None);
        assert_eq!(doc.get_path("a]"), None);
        assert_eq!(doc.get_path("a[]"), None);
        assert_eq!(doc.get_path("[a]"), None); // empty leading token: invalid
    }

    #[test]
    fn get_path_bracket_key_on_object() {
        // Bracketed tokens on an *object* are key lookups.
        let doc = obj(&[("a", obj(&[("0", Value::i64(5)), ("x", Value::i64(1))]))]);
        assert_eq!(doc.get_path("a[0]"), Some(&Value::i64(5)));
        assert_eq!(doc.get_path("a[x]"), Some(&Value::i64(1)));
        assert_eq!(doc.get_path("a[missing]"), None);
    }

    // -- set_path ----------------------------------------------------------------

    #[test]
    fn set_path_object_leaf() {
        let mut o = obj(&[("a", Value::i64(1))]);
        o.set_path("a", Value::i64(9)).unwrap(); // replace
        o.set_path("b", Value::str("x")).unwrap(); // insert
        assert_eq!(o, obj(&[("a", Value::i64(9)), ("b", Value::str("x"))]));

        // creates missing intermediates
        let mut empty = Value::object();
        empty.set_path("x.y.z", Value::i64(1)).unwrap();
        assert_eq!(empty, obj(&[("x", obj(&[("y", obj(&[("z", Value::i64(1))]))]))]));
    }

    #[test]
    fn set_path_array_leaf() {
        let mut a = arr(&[Value::i64(0), Value::i64(1)]);
        a.set_path("1", Value::i64(42)).unwrap(); // replace in range
        assert_eq!(a, arr(&[Value::i64(0), Value::i64(42)]));

        a.set_path("2", Value::i64(3)).unwrap(); // append at len
        assert_eq!(a, arr(&[Value::i64(0), Value::i64(42), Value::i64(3)]));

        assert_eq!(
            a.set_path("9", Value::i64(3)),
            Err(PathError::IndexOutOfRange { index: 9, len: 3 })
        );
        assert_eq!(a.set_path("x", Value::i64(3)), Err(PathError::NotAnIndex("x".into())));
    }

    #[test]
    fn set_path_creates_sparse_array_for_missing_field() {
        let mut o = Value::object();
        o.set_path("missing.2", Value::i64(5)).unwrap();
        assert_eq!(
            o,
            obj(&[("missing", arr(&[Value::Null, Value::Null, Value::i64(5)]))])
        );

        // deeper: missing.1.nested -> [Null, {nested: v}]
        let mut o2 = Value::object();
        o2.set_path("missing.1.nested", Value::i64(7)).unwrap();
        assert_eq!(
            o2,
            obj(&[("missing", arr(&[Value::Null, obj(&[("nested", Value::i64(7))])]))])
        );
    }

    #[test]
    fn set_path_intermediate_array_pad_and_extend() {
        let mut o = obj(&[("a", arr(&[Value::Null, Value::Null]))]);
        // pad to index 2 (== len) then set leaf
        o.set_path("a.2.x", Value::i64(1)).unwrap();
        assert_eq!(
            o,
            obj(&[("a", arr(&[Value::Null, Value::Null, obj(&[("x", Value::i64(1))])]))])
        );
        // index > len on an existing array is an error (no sparse extend)
        assert_eq!(
            o.set_path("a.9.x", Value::i64(1)),
            Err(PathError::IndexOutOfRange { index: 9, len: 3 })
        );
    }

    #[test]
    fn remove_path_object_and_nested() {
        let mut o = obj(&[("a", obj(&[("b", Value::i64(1)), ("c", Value::i64(2))])), ("x", Value::i64(9))]);
        // top-level leaf
        assert_eq!(o.remove_path("x"), Ok(true));
        assert_eq!(o, obj(&[("a", obj(&[("b", Value::i64(1)), ("c", Value::i64(2))]))]));
        // nested leaf
        assert_eq!(o.remove_path("a.b"), Ok(true));
        assert_eq!(o, obj(&[("a", obj(&[("c", Value::i64(2))]))]));
        // missing key: no-op, not an error
        assert_eq!(o.remove_path("a.nope"), Ok(false));
        assert_eq!(o.remove_path("nope"), Ok(false));
        assert_eq!(o, obj(&[("a", obj(&[("c", Value::i64(2))]))]));
    }

    #[test]
    fn remove_path_array() {
        let mut a = arr(&[Value::i64(0), Value::i64(1), Value::i64(2)]);
        assert_eq!(a.remove_path("1"), Ok(true)); // remove middle
        assert_eq!(a, arr(&[Value::i64(0), Value::i64(2)]));
        assert_eq!(a.remove_path("0"), Ok(true));
        assert_eq!(a, arr(&[Value::i64(2)]));
        // idx == len: nothing to remove (no-op, not an error)
        assert_eq!(a.remove_path("1"), Ok(false));
        // idx > len: out of range error
        assert_eq!(a.remove_path("5"), Err(PathError::IndexOutOfRange { index: 5, len: 1 }));
        // non-index token on an array
        assert_eq!(a.remove_path("x"), Err(PathError::NotAnIndex("x".into())));
    }

    #[test]
    fn remove_path_nested_array_of_objects() {
        let mut o = obj(&[("arr", arr(&[obj(&[("x", Value::i64(1))]), obj(&[("x", Value::i64(2))])]))]);
        assert_eq!(o.remove_path("arr.0.x"), Ok(true));
        assert_eq!(o, obj(&[("arr", arr(&[obj(&[]), obj(&[("x", Value::i64(2))])]))]));
        // idx 5 > len 2 at the leaf: out-of-range error
        assert_eq!(o.remove_path("arr.5"), Err(PathError::IndexOutOfRange { index: 5, len: 2 }));
        // missing intermediate object key: no-op
        assert_eq!(o.remove_path("nope.0.x"), Ok(false));
    }

    #[test]
    fn set_path_errors() {
        let mut v = Value::object();
        assert_eq!(v.set_path("", Value::i64(1)), Err(PathError::InvalidPath(String::new())));
        assert_eq!(v.set_path("a..b", Value::i64(1)).is_err(), true);
        assert_eq!(v.set_path("a[0]b", Value::i64(1)).is_err(), true);
        assert_eq!(v.set_path("a[", Value::i64(1)).is_err(), true);
        assert_eq!(v.set_path("a[]", Value::i64(1)).is_err(), true);
        // a stray `]` (no matching `[`) is just part of the key
        v.set_path("a]", Value::i64(1)).unwrap();
        assert_eq!(v.get_path("a]"), Some(&Value::i64(1)));

        // descending into a scalar
        let mut s = obj(&[("n", Value::i64(1))]);
        assert_eq!(
            s.set_path("n.x", Value::i64(2)),
            Err(PathError::CannotDescend { found: "i64" })
        );

        // numeric-looking key on an object is a key, not an index
        let mut k = Value::object();
        k.set_path("0", Value::i64(9)).unwrap();
        assert_eq!(k.get_path("0"), Some(&Value::i64(9)));
    }

    // -- Display ---------------------------------------------------------------

    #[test]
    fn display_scalars() {
        assert_eq!(Value::null().to_string(), "null");
        assert_eq!(Value::bool(true).to_string(), "true");
        assert_eq!(Value::bool(false).to_string(), "false");
        assert_eq!(Value::i64(42).to_string(), "42");
        assert_eq!(Value::i64(-7).to_string(), "-7");
        // F64 always shows a decimal point
        assert_eq!(Value::f64(1.0).to_string(), "1.0");
        assert_eq!(Value::f64(1.5).to_string(), "1.5");
        assert_eq!(Value::f64(-0.5).to_string(), "-0.5");
        assert_eq!(Value::f64(1e30).to_string(), "1000000000000000000000000000000.0");
        assert_eq!(Value::f64(f64::INFINITY).to_string(), "null");
        assert_eq!(Value::f64(f64::NEG_INFINITY).to_string(), "null");
        assert_eq!(Value::f64(f64::NAN).to_string(), "null");
        assert_eq!(Value::str("plain").to_string(), "\"plain\"");
    }

    #[test]
    fn display_string_escaping() {
        assert_eq!(Value::str("he said \"hi\"").to_string(), "\"he said \\\"hi\\\"\"");
        assert_eq!(Value::str("a\\b").to_string(), "\"a\\\\b\"");
        assert_eq!(Value::str("a\nb").to_string(), "\"a\\nb\"");
        assert_eq!(Value::str("a\rb").to_string(), "\"a\\rb\"");
        assert_eq!(Value::str("a\tb").to_string(), "\"a\\tb\"");
        assert_eq!(Value::str("a\0b").to_string(), "\"a\\u0000b\"");
        assert_eq!(Value::str("a\u{1}b").to_string(), "\"a\\u0001b\"");
        // unicode passes through unescaped
        assert_eq!(Value::str("moo🐄").to_string(), "\"moo🐄\"");
    }

    #[test]
    fn display_containers() {
        assert_eq!(Value::array().to_string(), "[]");
        assert_eq!(Value::object().to_string(), "{}");
        assert_eq!(
            arr(&[Value::i64(1), Value::bool(true), Value::null()]).to_string(),
            "[1, true, null]"
        );
        assert_eq!(
            obj(&[("a", Value::i64(1)), ("b", arr(&[Value::bool(true), Value::null()]))]).to_string(),
            "{\"a\": 1, \"b\": [true, null]}"
        );
        // insertion order preserved in output
        assert_eq!(obj(&[("z", Value::i64(1)), ("a", Value::i64(2))]).to_string(), "{\"z\": 1, \"a\": 2}");
        // nested
        assert_eq!(
            obj(&[("a", obj(&[("b", Value::i64(3))]))]).to_string(),
            "{\"a\": {\"b\": 3}}"
        );
    }

    // -- ordering & equality ---------------------------------------------------

    #[test]
    fn order_total_by_type_rank() {
        let values = vec![
            Value::Null,
            Value::bool(false),
            Value::bool(true),
            Value::i64(-1),
            Value::i64(1),
            Value::f64(1.5),
            Value::f64(2.5),
            Value::str("a"),
            arr(&[Value::i64(1)]),
            obj(&[("a", Value::i64(1))]),
        ];
        for i in 0..values.len() {
            for j in 0..values.len() {
                let c = values[i].cmp(&values[j]);
                let expect = if i < j { Ordering::Less } else { Ordering::Greater };
                if i != j {
                    assert_eq!(c, expect, "{} vs {}", values[i], values[j]);
                } else {
                    assert_eq!(c, Ordering::Equal);
                }
                assert_eq!(values[i].partial_cmp(&values[j]), Some(c));
            }
        }
    }

    #[test]
    fn cross_numeric_exact_compare() {
        let two_pow_53 = 1i64 << 53;
        // 2^53 + 1 is not representable in f64: the exact compare must not
        // collapse it to 2^53.
        assert_eq!(Value::i64(two_pow_53 + 1).cmp(&Value::f64(two_pow_53 as f64)), Ordering::Greater);
        assert_eq!(
            Value::i64(two_pow_53 + 1).cmp(&Value::f64((two_pow_53 + 2) as f64)),
            Ordering::Less
        );
        assert_eq!(Value::i64(two_pow_53).cmp(&Value::f64(two_pow_53 as f64)), Ordering::Equal);
        // negative side
        assert_eq!(
            Value::i64(-(two_pow_53 + 1)).cmp(&Value::f64(-(two_pow_53 + 2) as f64)),
            Ordering::Greater
        );
        // equality is cross-type
        assert_eq!(Value::i64(1), Value::f64(1.0));
        assert_eq!(Value::f64(-0.0), Value::i64(0));
        assert_ne!(Value::i64(1), Value::f64(1.5));
        assert_ne!(Value::i64(0), Value::bool(false)); // different type rank
        // tiny positive denormal > 0
        assert_eq!(Value::i64(0).cmp(&Value::f64(5e-324)), Ordering::Less);
        assert_eq!(Value::i64(-1).cmp(&Value::f64(-0.0)), Ordering::Less);
    }

    #[test]
    fn nan_orders_after_inf_and_eq_itself() {
        let nan = Value::f64(f64::NAN);
        let inf = Value::f64(f64::INFINITY);
        let neg_inf = Value::f64(f64::NEG_INFINITY);
        assert_eq!(nan.cmp(&nan), Ordering::Equal);
        assert_eq!(nan.cmp(&inf), Ordering::Greater);
        assert_eq!(nan.cmp(&neg_inf), Ordering::Greater);
        assert_eq!(inf.cmp(&nan), Ordering::Less);
        assert_eq!(nan, Value::f64(f64::NAN)); // Eq consistency

        // the I64/F64 band is shared: numerically equal values compare Equal
        assert_eq!(Value::i64(1).cmp(&Value::f64(1.0)), Ordering::Equal);
        assert_eq!(Value::f64(0.5).cmp(&Value::i64(1)), Ordering::Less);
    }

    #[test]
    fn object_equality_and_order_are_canonical() {
        let o1 = obj(&[("a", Value::i64(1)), ("b", Value::str("x"))]);
        let o2 = obj(&[("b", Value::str("x")), ("a", Value::i64(1))]);
        assert_eq!(o1, o2); // key order irrelevant to equality
        assert_eq!(o1.cmp(&o2), Ordering::Equal);

        // canonical key order drives comparison
        let c = obj(&[("a", Value::i64(1)), ("b", Value::i64(2))]);
        let d = obj(&[("a", Value::i64(2))]);
        assert_eq!(c.cmp(&d), Ordering::Less); // a:1 < a:2
        let e = obj(&[("a", Value::i64(1)), ("c", Value::i64(0))]);
        assert_eq!(c.cmp(&e), Ordering::Less); // b < c

        // arrays are element-wise, not canonical
        assert_eq!(arr(&[Value::i64(1), Value::i64(2)]).cmp(&arr(&[Value::i64(1), Value::i64(3)])), Ordering::Less);
        assert_eq!(arr(&[Value::i64(1)]).cmp(&arr(&[Value::i64(1), Value::i64(0)])), Ordering::Less);
    }

    #[test]
    fn nested_equality() {
        let d1 = obj(&[("a", arr(&[Value::i64(1), Value::f64(2.0)])), ("b", Value::i64(3))]);
        let d2 = obj(&[("b", Value::i64(3)), ("a", arr(&[Value::f64(1.0), Value::i64(2)]))]);
        assert_eq!(d1, d2);
        let d3 = obj(&[("b", Value::i64(3)), ("a", arr(&[Value::i64(1), Value::i64(3)]))]);
        assert_ne!(d1, d3);
    }

    // -- parse_path edge coverage ----------------------------------------------

    #[test]
    fn parse_path_edge_cases() {
        assert!(parse_path("").is_err());
        assert!(parse_path(".").is_err());
        assert!(parse_path("..").is_err());
        assert!(parse_path("a.").is_err());
        assert!(parse_path(".a").is_err());
        assert!(parse_path("a..b").is_err());
        assert!(parse_path("a[0]b").is_err());
        assert!(parse_path("a[0].b[1]c").is_err());
        assert!(parse_path("a[").is_err());
        assert!(parse_path("a[]").is_err());
        assert!(parse_path("a[b].").is_err());
        assert!(parse_path("[a]").is_err()); // path must start with a bare key

        // a bracket token at the *end* of a path is fine
        let toks = parse_path("a[0]").unwrap();
        assert_eq!(toks.as_slice(), ["a", "0"]);

        // chained brackets are fine
        let toks = parse_path("a[0][1]").unwrap();
        assert_eq!(toks.as_slice(), ["a", "0", "1"]);
        assert!(parse_path("a[0]x[1]").is_err()); // junk between brackets
        assert!(parse_path("a.[1]").is_err());    // bracket right after `.`

        // a stray `]` without `[` stays inside the bare key
        assert_eq!(parse_path("a]b").unwrap().as_slice(), ["a]b"]);

        let toks = parse_path("a[0].b[1]").unwrap();
        assert_eq!(toks.as_slice(), ["a", "0", "b", "1"]);

        // keys may contain brackets/dots only if balanced and bare
        let toks = parse_path("a.b-c_d").unwrap();
        assert_eq!(toks.as_slice(), ["a", "b-c_d"]);
    }
}
