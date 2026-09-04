# spec.md — MooRacer

An in-memory, network-accessible document data engine in **pure Rust**. No disk,
no persistence, no durable state of any kind — think "Redis but a document store
and entirely in RAM." It provides the full query surface of [Moofile]
(https://github.com/patw/moofile) **except auto-embedding / semantic search**
(voyage-4-nano ONNX inference is out of scope). Everything lives in memory and
every operator is hyper-optimized for raw throughput and low latency.
"Safety is not a concern, only raw performance" — correctness is still expected
(failing tests are a bug), but the code should aggressively optimize hot paths,
including measured `unsafe` where it pays off.

The shipped build targets a **CPU-native release profile** (fat LTO, single
codegen unit, `-C target-cpu=native`) — the flag set the benchmark harness and
server are measured under.

## Non-goals
- No persistence / no disk I/O / no durability / no recovery.
- No auto-embedding, no ONNX, no semantic search, no inference of any kind.
- No cross-process file locking (single process, in-memory). Thread-safe in one
  process: concurrent reads, serialized writes.
- No MongoDB compatibility layer or BSON file format. Documents are in-memory
  native value trees.

## Data model
- **Collection**: a named in-memory store of documents.
- **Document**: an arbitrary nested JSON-like value tree (objects, arrays, scalars).
  Keyed by string `_id` (unique, always present, auto-generated as a
  24-char hex string if absent, must be a string; cannot be changed by update).
- **Value model**: a compact native enum (`Null | Bool | I64 | F64 | String |
  Array | Object`) with a fast path-based get/query evaluator. Optimize the
  internal representation (e.g. inline small scalars, arena/boxed objects) for
  speed and cache-friendliness; do not require `serde_json::Value` unless it
  benchmarks as competitive.

  *Value model decisions* (fixed during implementation; authoritative for the
  query/index subtasks — see `engine/src/value.rs`):
  - Path syntax for `get_path`/`set_path`: `.` and `[...]` separators are
    interchangeable (`a.b.c`, `a[0].b`, `a.b[2][3]`). A token is **always a
    key** on an object (numeric keys like `"0"` work) and must be an unsigned
    decimal index on an array.
  - `set_path`: leaf replaces an in-range array slot, appends at `len`,
    creates missing intermediate objects, and pads missing intermediate array
    fields with `Null` (sparse create); index > len on an existing array or
    descending into a non-null scalar is an error.
  - Total order used by indexes and `.sort()`:
    `Null < Bool < Number < Str < Array < Object`; `I64`/`F64` compare
    **exactly** across types (`1 == 1.0`, no 2^53 precision collapse);
    objects compare canonical (key order irrelevant), matching Mongo document
    equality; `NaN` is total (equals itself, orders after `+inf`).
  - `Display` is JSON-like logging output, **not** a wire codec (the wire
    format is FlatBuffers): `F64` always shows a decimal point
    (`1.0`), non-finite floats render as `null`, strings are JSON-escaped.

## Query API (Mongo-style chain)
Expose a fluent query builder that is evaluated lazily:
`.find(filter)`, `.find_one(filter)`, `.count(filter)`, `.exists(filter)`,
`.sort(field, descending)`, `.skip(n)`, `.limit(n)`, terminal `.to_list()/
.first()/.count()`.

Filter operators (must all work):
- Comparison: implicit `$eq`, plus `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, and
  ranges (`{"age": {"$gte":25, "$lt":40}}`).
- Set: `$in`, `$nin`.
- Logical: `$and`, `$or`, `$not`.
- Element: `$exists`.
- Array: `$elemMatch` (equality, comparison, and sub-document filters).

An index is used automatically when the filter's top-level field is indexed;
otherwise a full scan. `_id` is always indexed.

  *Query builder decisions* (fixed during implementation; authoritative for the
  operator/search subtasks — see `engine/src/query.rs`):
  - `Query` is lazy: it borrows `&Collection` and owns the filter `Value`; the
    single scan runs only at a terminal (`.to_list()`/`.first()`/`.count()`, and
    later `.sort`/`.skip`/`.limit`). `find`/`find_one`/`count`/`exists` are the
    eager entry points; a terminal consumes its `Query`.
  - The filter is a `Value` **object** in Mongo syntax. `{}` (the empty object)
    matches every document; a **non-object** filter is malformed and matches
    nothing (defensive, no panic).
  - Top-level keys combine as an implicit `$and`. A condition that is an
    **operator object** (a non-empty object whose keys *all* start with `$`,
    e.g. `{"$gte":25}`) dispatches to the operator matchers (comparison /
    set / logical subtasks); a condition with a mixed `$`-and-plain key set is
    *not* an operator object. Any other condition is a **direct value**
    (implicit `$eq`) matched by **exact canonical `Value` equality**.
  - A direct-value match on a nested object is an **exact subdocument
    equality** (MongoDB semantics), *not* a subset match: an extra field on
    the stored side disqualifies the document (`{addr:{city:"NYC"}}` does not
    match a doc whose `addr` also has `zip`).
  - Result order is the collection's storage order until `.sort()`; an
    index-driven scan returns matching docs in **index order** (field value
    ascending per the total order, ties by `_id`) — see the comparison-
    operator decisions below.

  *Query pipeline decisions* (fixed during implementation; authoritative for
  the aggregation/search subtasks — see `engine/src/query.rs`):
  - `.sort(field, descending)`, `.skip(n)`, `.limit(m)` form the Mongo
    pipeline **filter → sort → skip → limit**, applied identically by every
    terminal (`.to_list()`, `.first()`, `.count()`). A query has at most one
    sort field (a later `.sort` replaces the earlier one).
  - Sort order is the engine **total order** on the field value; ties break
    by `_id`. `descending = true` reverses the **whole** (value, `_id`)
    order (i.e. it is exactly the ascending order reversed, so equal-value
    ties come back `_id`-descending). A **missing** sort field sorts like
    `Null` (first in ascending, last in descending — the index's missing-
    field convention).
  - `.skip(n)` drops the first `n` docs of the (sorted) filtered stream;
    `.limit(m)` returns at most `m` docs; **`limit(0)` means no limit** (the
    Mongo cursor convention). Without a sort, skip/limit apply in scan order
    (storage order for a full scan, index order for an index-driven scan).
  - **Sort fast path**: when the sort field is **indexed**, the terminal
    streams the index itself in (reverse) order — a double-ended walk of the
    contiguous index array (no allocation, no sort step) — verifying each
    entry against the full filter; the walk stops after the last doc returned
    by `skip`/`limit`, so `sort + limit` never materializes more than
    `skip + limit` matching docs. With an **unindexed** sort field the
    matches are collected and sorted (total-order comparator: field value,
    then `_id`); indexed and unindexed sorts always produce identical
    results.

  *Comparison operator decisions* (fixed during implementation; authoritative
  for the set/logical/element/array operator subtasks — see
  `engine/src/query.rs`):
  - An operator object dispatches to `$eq`, `$ne`, `$gt`, `$gte`, `$lt`,
    `$lte`; several operators on one field are **AND-ed** (range combos,
    e.g. `{"age": {"$gte": 25, "$lt": 40}}`). Values compare with the engine
    **total order** (exact cross-numeric, total NaN; ranks
    `Null < Bool < Number < Str < Array < Object`).
  - **Missing-field rules** (MongoDB): `$gt`/`$gte`/`$lt`/`$lte` require the
    field to be present; `$ne` matches a missing field unless its operand is
    `null`; `$eq` matches a missing field only when its operand is `null`
    (so `{$eq: null}` and the direct value `null` both match explicit null
    *and* absence).
  - **No array containment** in this operator family: a field that is an
    array is compared as a whole value in the total order. Element-level
    array matching is `$elemMatch`'s territory (a later subtask).
  - An **unknown `$` operator makes the condition match nothing** (no error
    channel in the lazy `Query`; malformed filters never panic) and cannot
    drive an index scan.
  - **Index-driven scans**: when a top-level condition is on an indexed field
    and carries at least one bound (`$eq`/`$gt`/`$gte`/`$lt`/`$lte`, or a
    direct value), the terminal fetches the candidate id range from that
    field's index (a direct value is the point range; `$gt`/`$lt` give
    `Excluded` bounds, `$gte`/`$lte`/`$eq` give `Included` ones, same-side
    bounds tighten to the stricter, exclusive wins ties) and re-verifies
    every candidate against the full filter — index-driven and full-scan
    results are always identical. A bare `$ne` (or a condition with an
    unknown operator) never drives a scan: the plain scan is just as good.
    The first indexable condition in filter entry order is the driver;
    remaining conditions verify on the candidates.
  - For a driver on a field whose documents may lack the field, the index's
    `Null` entries (missing *and* explicit null) may appear in the candidate
    range; the presence rules above are what filter them during verification.

  *Set operator decisions* (fixed during implementation; authoritative for
  the logical/element/array operator subtasks — see `engine/src/query.rs`):
  - `$in` is an **OR of `$eq` over its array operand**: the field's **whole
    value** equals some list element under the engine total order (exact
    cross-numeric, total NaN, canonical objects). **No array containment** —
    a stored array matches only a list element that is exactly that array.
    List order and duplicates are irrelevant.
  - **Missing-field rules**: a *missing* field matches `$in` only when the
    list contains `null` (the inherited `$eq` rule), so `{$in: [null]}`
    behaves like `{$eq: null}` (explicit null *and* absence). `$nin` is the
    **exact complement** of `$in`. An **empty list**: `$in` matches nothing,
    `$nin` matches everything (missing included).
  - A **non-array `$in`/`$nin` operand** makes the whole condition match
    nothing (defensive — malformed filters never panic) and cannot drive an
    index scan.
  - **Index-driven scans**: an `$in` drives as the **union of its list's
    point ranges** — the list is deduped and walked in total-order ascending
    so results come back in index order (value ascending per the total
    order, ties by `_id`); an empty list is the empty candidate set (the
    terminal short-circuits without touching the collection). Other
    operators in the same condition (`$ne`, bounds) verify on the
    candidates. A bare `$nin` never drives (it would return almost
    everything); a `$nin` alongside a bound is verified after the bound's
    range drives.

  *Logical operator decisions* (fixed during implementation; authoritative
  for the element/array operator subtasks — see `engine/src/query.rs`):
  - `$and`/`$or` are **top-level-only** operators whose operand is an
    **array of sub-filters** (filter objects), each evaluated with the full
    filter semantics: `$and` requires **all** to match (an **empty list is
    vacuous truth** — matches every document), `$or` requires **at least
    one** (an **empty list matches nothing**). Elements are sub-filters, so
    the operators **nest** (`{$or: [{$and: [...]}]}`) and every other
    operator works inside them.
  - `$not` is **field-level**: `{"f": {$not: {<operator expression>}}}` is
    the negation of the whole expression (all its operators AND-ed,
    missing-field/presence rules included) — `{$not: {$gt: 5}}` is "f ≤ 5
    *or f missing*"; it ANDs with other operators on the same field like
    any other operator.
  - **Malformed shapes match nothing, never panic** (the lazy `Query` has no
    error channel): a non-array `$and`/`$or` operand, a non-object `$and`
    element, a non-operator-object `$not` operand (direct value, plain
    object, array, empty object), a field-level `$and`/`$or`
    (`{"f": {$and: [...]}}`), and a **top-level `$not`** all match nothing.
    A non-object element inside `$or` fails only its own disjunct.
  - Top-level `$and`/`$or` keys are **reserved**: they are not matched as
    direct values against a literal `$and`/`$or` document field.
  - **Index-driven scans**: entry order walks top-level keys first, then —
    because a conjunct narrows the result — the elements of a top-level
    `$and` (looked through recursively); the first indexable condition
    among them drives as usual. A top-level `$or` never drives (it is a
    union; no single condition's candidate set contains it), a `$not`
    contributes no bound (verification only), and a malformed `$and`
    operand/element cannot drive. Every candidate is still re-verified
    against the full filter, so index-driven and full-scan results are
    always identical.

  *Element operator decisions* (fixed during implementation; authoritative
  for the array-operator subtask — see `engine/src/query.rs`):
  - `$exists` is **presence-only** and ignores the field's value:
    `{"f": {$exists: true}}` matches a document whose `f` key **exists**
    (an explicit `null` *counts as present*), and `{"f": {$exists: false}}`
    matches a **missing** key. The two are exact complements and partition
    every document.
  - The operand must be a **boolean**; a non-boolean `$exists` operand makes
    the whole condition match nothing (defensive — malformed filters never
    panic).
  - `$exists` **never drives an index scan**: a field index stores both a
    missing field and an explicit `null` as the single `Null` entry, so the
    index cannot distinguish presence from absence. A `$exists` condition is
    always verified on the full scan; a sibling bound (e.g. `$gte`) still
    drives as usual and `$exists` verifies the candidates.

  *Array operator decisions* (fixed during implementation; authoritative for
  later search/aggregation subtasks — see `engine/src/query.rs`):
  - `$elemMatch` (`{"f": {$elemMatch: {<criteria>}}}`) matches a document when
    **at least one** element of the array field `f` satisfies `<criteria>`.
    The field must be present **and** an array: a missing field or a non-array
    (scalar) field has no elements and matches nothing for *any* criteria.
  - The `<criteria>` operand is classified once and applied per element:
    - a **direct value** (scalar, or a value that is not an operator object)
      → **element equality** (an element equals the operand under the engine
      total order — cross-numeric, total NaN, canonical objects);
    - an **operator object** (non-empty object whose keys all start with `$`,
      e.g. `{$gt: 5}`, `{$gte: 1, $lt: 10}`) → the element (always present)
      must satisfy all its operators; the missing-field rules degenerate
      because every element is present, so `$eq`/`$ne` are plain equality and
      the comparison operators are plain total-order comparisons;
    - a **sub-document** (a non-operator object, e.g.
      `{qty: {$gt: 5}, warehouse: "A"}`) → the **full filter** is evaluated
      against each element, which is treated as a document (nested operators
      and further sub-documents work inside).
  - `$elemMatch` is **field-level** (as with `$not`/`$exists`): it is
    dispatched from the operator-object condition of a field and works inside
    `$and`/`$or` sub-filters. It is *not* a reserved top-level keyword like
    `$and`/`$or`; a top-level `{"$elemMatch": ...}` is treated as an ordinary
    field query (matching a literal `"$elemMatch"` document field) and is not
    the array operator.
  - `$elemMatch` **never drives an index scan**: a field index stores the
    whole array as a single `Value` entry (elements are not indexed), so the
    index cannot split candidates by element. A `$elemMatch` condition
    contributes no bound and is verified on the (full) scan, exactly like
    `$exists`/`$ne`/`$nin`/`$not`; a sibling bound on a *different* top-level
    field still drives as usual.

## Write API
- `insert(doc)` / `insert_many(docs)` — assigns `_id` if missing; duplicates
  raise an error.

  *Document store decisions* (fixed during implementation; authoritative for
  the index/query subtasks — see `engine/src/collection.rs`):
  - `insert`/`insert_many` reject non-object documents (`NotAnObject`).
  - `insert_many` is **atomic (staged)**: the whole batch is validated — doc
    shape and `_id` uniqueness against the store *and* within the batch —
    before any doc commits; a rejected batch leaves the collection unchanged.
  - Auto-generated `_id`: 24-char zero-padded lowercase hex from a
    process-wide atomic counter (not random); unique across all collections
    in the process.
  - A generated `_id` is prepended as the **first** key; a user-supplied
    `_id` keeps its position (docs are not reordered).
  - `_id` uniqueness is per-collection: explicit ids may repeat across
    different collections.
- `update_one(filter, set=...)` / `update_many(filter, set=...)` — `set`,
  `inc`, `unset` update operators.
- `replace_one(filter, new_doc)` — preserve `_id`.
- `delete_one(filter)` → bool; `delete_many(filter)` → count.

  *Replace decisions* (fixed during implementation; see `engine/src/collection.rs`):
  - `replace_one` matches **the first** document in storage order (like
    `update_one`) and replaces it **wholesale**: fields absent from `new_doc`
    are dropped (this is a replacement, not an update). It returns the match
    count (`1` on success) and is `Err(StoreError::NoMatch)` when nothing
    matches (the same convention as `update_one`).
  - The matched document's `_id` is **always preserved**: a `new_doc` whose
    explicit string `_id` differs is `Err(StoreError::IdMismatch)` (store
    untouched); a `new_doc` with no `_id` has the matched `_id` restored as
    the first key; a non-object `new_doc` is `Err(StoreError::NotAnObject)`.
    All cases run through the `set_doc` primitive, so every field index is
    deindexed (old) and reindexed (new) in lockstep (missing fields → `Null`).
- **Atomic batch**: a transaction context where all writes in the batch apply
  atomically on commit; reads inside see the pre-batch state; rollback on
  error. Batch writes update all indexes together.
- Index maintenance must be correct on insert/update/delete (add/remove/refresh
  entries), or indexes rebuilt deterministically.

  *Update operator decisions* (fixed during implementation; authoritative for
  the `replace_one`/`delete_*`/atomic-batch subtasks — see
  `engine/src/collection.rs`):
  - The `update` argument to `update_one`/`update_many` is a Mongo operator
    object: `{"$set": {…}, "$inc": {…}, "$unset": {…}}`. Operators apply in
    the order they appear in the spec (so `{$set:{f:10}, $inc:{f:3}}` = 13;
    reversed = 10). An empty operator object is a valid no-op update.
  - **`$set`** uses `set_path` semantics (replace in-range array slot,
    append at `len`, create missing intermediate objects, pad missing
    intermediate arrays with `Null`). **`$inc`** adds a numeric operand to
    the field (a missing field is created with the operand; an existing
    non-numeric field is an error; `i64+i64` stays `i64` unless it overflows
    `i64`, in which case it widens to `F64`; any float operand/value yields
    `F64`). **`$unset`** removes paths; its operand is either an object (keys
    used, values ignored) or an array of string field names; removing a
    missing path is a no-op.
  - **`_id` cannot be changed**: a `$set`/`$inc`/`$unset` naming `_id` is an
    error (`StoreError::InvalidUpdate`), and the stored `_id` is preserved and
    un-reordered on every update.
  - `update_one` returns the match count (always `1` on success) and is
    **`Err(StoreError::NoMatch)`** when nothing matches (the spec's "errors on
    no-match" — `update_many` instead returns the count, `0` valid, no error).
    A malformed spec (non-object spec, unknown operator, non-object
    `$set`/`$inc` operand, non-string `$unset` array element) is
    **`StoreError::InvalidUpdate`** and changes nothing. `update_many`
    shape-validates the spec **before touching any document**, so a malformed
    spec is applied to none.
  - **Index maintenance**: updates run through `Collection::set_doc`
    (deindex old + index new), so every field index stays in lockstep; a
    field index on an updated field reflects the new value immediately.

  *Atomic batch / transaction decisions* (fixed during implementation;
  authoritative for the server/client write-path subtasks — see
  `engine/src/collection.rs`, `Collection::begin()` → `Transaction`):
  - `Collection::begin()` returns a `Transaction<'_>` that borrows the
    collection. It **buffers** writes; the live store is untouched until
    `commit()`. `commit()` applies the whole batch (docs **and every field
    index**) as one unit; `rollback()` — or simply dropping the transaction
    without committing — discards all buffered writes. Dropping without
    commit is a rollback because no write ever mutates the collection before
    commit (no `Drop` hook needed).
  - **Reads inside the transaction see the pre-batch state**: `find` /
    `find_one` / `count` / `get` / `contains` / `len` / `index` all read the
    live (unmodified) store, so staged writes are invisible until commit.
  - **Writes are id-scoped concrete mutations** resolved against the
    pre-batch snapshot: an insert is a `Put` of a new id; `update(filter,
    spec)` (≡ `update_many`) and `replace(filter, doc)` (≡ `replace_one`)
    produce a full new doc (`Put`) per matched id; `delete(filter)` (≡
    `delete_many`) is a `Delete` per matched id. Batch writes therefore do
    NOT compose sequentially against each other — every op reads the same
    pre-batch snapshot (e.g. two `$inc +1`/`+5` on a field both compute from
    the pre-batch value, and the later op wins).
  - **Per-id composition** on the batch buffer: a later write to an id
    already written replaces the earlier op (each id commits at most once).
    `delete` on an id that carries a staged `Put`: if the id is a **pre-batch
    doc**, the put becomes a `Delete` (net removal); if the id was **only
    inserted in this batch**, the put is removed entirely (net no-op).
    Because reads see pre-batch state, a batch-inserted id cannot be targeted
    by a filter-based `update`/`replace`/`delete` in the same batch.
  - **Rollback on error**: any write that errors — a duplicate `_id` (against
    the pre-batch store *or* an earlier batch op), a non-object doc, an
    `_id`-must-be-string doc, an `_id` mismatch on `replace`, a malformed
    update spec, or a doc-dependent `$inc` error — marks the transaction
    *failed* (`is_failed()`, `error()`); a failed transaction's `commit()`
    applies nothing and returns the stored error. `update`/`replace`
    no-match (`NoMatch`) also fails the transaction.
  - Return conventions mirror the eager APIs: `insert` → `Result<String,_>`
    (the id), `insert_many` → `Result<usize,_>`, `update` → `Result<usize,_>`
    (matched count), `replace` → `Result<usize,_>`, `delete` → `usize`
    (matched count, never an error). `commit()` → `Result<(), StoreError>`;
    `rollback()` → `()`.

  *Value `remove_path` decisions* (the inverse of `set_path`, used by
  `$unset` — see `engine/src/value.rs`):
  - On an `Object` the key at `path` is removed; on an `Array` the element at
    the index is removed (later elements shift down). Returns `true` when
    something was removed, `false` on a no-op.
  - A **missing** intermediate step (key or index that does not exist) is a
    no-op (`Ok(false)`), not an error. Only a structurally invalid path, an
    out-of-range array index at the **leaf** (`index > len`; `index == len`
    is a no-op), a non-index token on an array, or descending into a scalar is
    a `PathError`.

## Indexes (all in memory)
- **Primary**: `_id` always.
- **Field indexes**: top-level field → ordered map for equality + range scans.

  *Index decisions* (fixed during implementation; authoritative for the
  query/search subtasks — see `engine/src/index.rs`):
  - An index is a sorted array of `(value, _id)` entries, ordered by
    (`Value`'s total order — exact cross-numeric, total NaN —, then `_id`
    byte order). Equality = one binary-search slice; range scans take
    `std::ops::Bound`s (`Included`/`Excluded`/`Unbounded`) and are two
    `partition_point`s. Equal values are returned in `_id` order, so
    index-based queries are deterministic.
  - The primary `_id` index always exists and cannot be dropped
    (`StoreError::PrimaryIndex`). Field indexes are created explicitly
    (`create_index`, backfilled from current docs; calling it again rebuilds
    deterministically) and dropped with `drop_index` (`NoIndex` when absent).
  - Every document has exactly one entry in every field index: a **missing
    field is indexed as `Null`** (MongoDB convention — `{"f": null}` matches
    both explicit `null` and absence), so index scans and full scans agree.
  - Maintenance primitives keep all indexes in lockstep with the store:
    `insert`/`insert_many` register entries; `remove_doc(id) -> Option<Value>`
    is the delete primitive (drops every entry); `set_doc(id, doc) ->
    Result<Option<Value>, _>` is the update/replace primitive (preserves
    `_id`, `IdMismatch` on a changed `_id`, `Ok(None)` no-op when the id is
    absent — filter-matches-first). `update_*`/`replace_one`/`delete_*`
    build on these.
  *Stats & reindex decisions* (fixed during implementation; see
  `engine/src/collection.rs` — `CollectionStats` / `IndexStats`):
  - `stats() -> CollectionStats` is a synchronous snapshot: `docs` (document
    count), `indexes` (index count, always `>= 1` for the primary), and
    `per_index: Vec<IndexStats>` sorted by field name (deterministic; always
    includes `_id`), each with `entries` (one per stored doc — missing field
    → `Null` entry), `distinct` (engine total order: `I64(1)` == `F64(1.0)`),
    and a capacity-based `memory` byte estimate. `docs_memory` estimates the
    doc store (map + id strings + recursive value trees) and
    `total_memory == docs_memory + sum(per_index.memory)` — an invariant the
    server layer can rely on.
  - `reindex() -> usize` deterministically rebuilds **every** index
    (including the primary `_id` index) from the current documents via a
    fresh sorted pass (`FieldIndex::load` — no incremental `insert`
    memmoves), and returns the number of indexes rebuilt. It is the
    consistency-safety operation: after any sequence of writes the rebuilt
    indexes are byte-for-byte equivalent to incremental maintenance, so
    `stats()` is identical before and after `reindex()`.
- **Vector index**: `field -> Vec<f32>` with configurable dimension; brute-force
  cosine similarity search, SIMD where beneficial, returning `(doc, score)`.
- **Text index**: BM25 full-text with Porter stemming on an inverted index,
  returning `(doc, score)`.

  *Vector index decisions* (fixed during implementation; authoritative for the
  search/hybrid subtasks — see `engine/src/vector.rs`):
  - `create_vector_index(field, dim)` registers a `field -> Vec<f32>` index
    with a fixed, configured `dim`, backfilling from the documents currently
    stored. It is a **separate index layer** from the field/`_id` index set
    (it does not appear in `stats()`'s `per_index`, which is about ordered
    `Value` indexes). `drop_vector_index` / `has_vector_index` /
    `vector_index` / `vector_index_names` are the accessors.
  - A document's field is a **top-level `Value::Array` of numbers** (`I64`/`F64`
    both coerce to `f32`). A **missing** field means the doc is simply **not
    indexed** (nothing to search; not an error). A **present** field that is not
    a `dim`-length numeric array (wrong length, a non-numeric element, or a
    non-array) is a write error `StoreError::VectorDimMismatch { field,
    expected, found }` — the write is rejected and the store is untouched
    (checked by `insert`, `insert_many`, `set_doc`, and `Transaction::insert`).
  - **Storage normalizes once at write time**: each stored vector is
    unit-normalized on insert, so a search normalizes only the query and every
    score is one `f32` dot product of two unit vectors (the cosine). This is
    the brute-force perf win: no per-query per-doc normalization or division.
    The dot product is a plain `zip`/`sum` over a contiguous `dim`-strided flat
    buffer and autovectorizes to SIMD under `target-cpu=native` + release.
  - `vector_search(field, query, limit) -> Result<Vec<(Value, f32)>, _>`:
    **brute-force cosine** over the index, returning the top `limit` documents
    (full clones) with their cosine score in `[-1, 1]`, best (most similar)
    first; ties break by `_id` ascending (stable by index order). `limit(0)`
    means no limit (return every indexed doc, best-first) — the same `0`
    convention as the query pipeline. A **wrong-length query** (≠ `dim`) or an
    **empty index** returns an empty vec (not an error); a **missing index**
    for `field` is `StoreError::NoIndex`. A **zero** vector (query or stored)
    scores `0.0` against everything (its norm is 0).
  - Search methods are **top-level on `Collection`** (they read an index, not a
    `find` filter), so they are not part of the `.find(...)` filter chain — the
    spec's `.vector_search(...)` notation denotes the Collection entry point.

  *Text index decisions* (fixed during implementation; authoritative for the
  hybrid-search subtask — see `engine/src/text.rs`):
  - `create_text_index(field)` registers a BM25 text index on a **top-level
    field** (a separate index layer from the `Value`/`_id` index set and the
    vector index; it does not appear in `stats()`). `drop_text_index` /
    `has_text_index` / `text_index` / `text_index_names` are the accessors.
    Calling it again deterministically rebuilds from the current docs.
  - A document's field is a `Value::Str` or a `Value::Array` of strings;
    **anything else — including a missing field and an array with a
    non-string element — is simply not indexed**. A text index **never
    rejects a write** (unlike the vector index, which enforces a dimension),
    so `insert`/`set_doc`/`Transaction` need no vector-style validation.
  - **Tokenization** is one allocation-light pass: tokens are maximal
    `[a-z0-9]` runs after lowercasing (non-ASCII bytes split the run).
    **Porter stemming** (the classic 1980 algorithm, byte-exact with the
    reference / NLTK `ORIGINAL_ALGORITHM`) is applied to every token at write
    time and to query tokens at search time, so the postings table only holds
    stems.
  - **Inverted layout**: `postings: HashMap<stem, Vec<(doc_idx, tf)>>` over
    parallel `ids`/`doc_lens` arrays (insertion order), so a search walks only
    the query stems' posting lists (no full corpus scan). `doc_lens` +
    `total_tokens` give BM25 length normalization with no per-query corpus
    tokenization.
  - **BM25 (Okapi)**, Lucene conventions: `k1 = 1.2`, `b = 0.75`,
    `idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))` (always positive),
    `score += idf · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`. The query is
    a **bag of terms**: each distinct query stem counts once. Only documents
    with a **strictly positive** score are returned.
  - `text_search(field, query, limit) -> Result<Vec<(Value, f64)>, StoreError>`
    returns the top `limit` documents (full clones) with their BM25 score,
    best (highest) first; ties break by index order (deterministic).
    `limit == 0` means no limit (the same `0` convention as the query pipeline
    and vector search). An **empty index** or a **tokenless query** returns an
    empty vec (not an error); a **missing index** for `field` is
    `StoreError::NoIndex`. Search is **top-level on `Collection`**, not part of
    the `.find(...)` filter chain.

## Search
- `Collection::vector_search(field, query_vec, limit)` — brute-force cosine.
- `.text_search(field, query, limit)` — BM25 + Porter.
- `.hybrid_search(text_field, vec_field, query_text, query_vec, limit)` —
  reciprocal-rank-fusion (RRF) of the BM25 and vector rankings.

  *Hybrid search decisions* (fixed during implementation; authoritative for the
  server/client search subtasks — see `Collection::hybrid_search`):
  - `hybrid_search(text_field, vec_field, query_text, query_vec, limit) ->
    Result<Vec<HybridHit>, _>` with `HybridHit = (Value, f64)`. Both indexes
    must exist: a missing **text** index on `text_field` or a missing **vector**
    index on `vec_field` is `StoreError::NoIndex`.
  - **Reciprocal Rank Fusion** over **ranks, not raw scores**: a document at
    1-based rank `r` in a ranking earns `1 / (RRF_K + r)` (the classic RRF
    constant `RRF_K = 60`), and its fused score is the **sum** over every
    ranking it appears in. Both full (unlimited) rankings are computed first,
    so the final `limit` is applied only *after* fusion (a truncated sub-ranking
    would corrupt the rank positions).
  - The fusion is a **union** over the two document sets: a document ranked by
    only one signal still surfaces (with that signal's contribution alone); a
    document in neither ranking is absent. The two signals need not agree.
  - Results are returned best (highest fused score) first, **ties by `_id`
    ascending** (deterministic); documents are returned as **full clones**.
    `limit == 0` means no limit (the same `0` convention as the single
    searches); an empty fusion returns an empty vec (not an error).
  - Search is **top-level on `Collection`**, not part of the `.find(...)`
    filter chain.

## Aggregation
`.find(filter).group(field).agg(...)` with `count`, `sum`, `mean`, `min`,
`max`, `collect`, `first`, `last` per group, plus optional `.sort`/`.limit`
on the grouped result.

### Aggregation decisions (contract)
- API: `Query::group(field) -> GroupQuery`, terminal
  `GroupQuery::agg(fn: AggFn, field) -> Vec<Value>`; `GroupQuery::sort(field,
  desc)` / `GroupQuery::limit(n)` apply to the **group documents** (the
  query's own sort/skip/limit apply to the *document* stream before grouping
  and define first/last/collect order within a group).
- Grouping key = the document's group-field value; a **missing group field
  groups under `Null`** (the missing-field convention). `I64(n)` and
  `F64(n.0)` are the same group (total-order equality).
- Result document per group: `{ "_id": <group key>, "<fn-name>": <result> }`
  (fn-names: `count`/`sum`/`mean`/`min`/`max`/`collect`/`first`/`last`).
- `count` ignores its `field` argument; result is `I64`.
- `sum`/`mean` take **numeric** values (`I64`/`F64`); missing or non-numeric
  values are skipped. All-`I64` sums stay `I64` and widen to `F64` only on
  overflow (the `$inc` rule, accumulated in i128); any `F64` operand makes
  the sum `F64`. `sum` over no numerics = `I64(0)`; `mean` over no numerics
  = `Null`, otherwise `F64`.
- `min`/`max` take the total-order extreme over **present** values (missing
  skipped); no present values = `Null`.
- `collect` appends one element per member **in stream order** (missing
  field contributes `Null`); `first`/`last` take the field value of the
  first/last member in stream order (missing = `Null`).
- Group order: default = group-key total order (deterministic);
  `GroupQuery::sort` re-sorts the group documents (total order, missing =
  `Null`, ties by `_id`); `limit(0)` = no limit (the Mongo convention).

## Network layer (FlatBuffers, high-throughput)
- Wire protocol: commands + replies encoded with **FlatBuffers** (the
  `flatbuffers` crate in Rust; the `flatbuffers` Python package client-side),
  length-prefixed frames over a binary TCP stream. Define one `.fbs` schema for
  the request/response envelope (a command union + typed payloads + status).
- **Server**: TCP listener, small thread pool (configurable), concurrent reads,
  serialized writes (RwLock); per-connection request/response loop. Fast path
  avoids allocations. Versioned envelope for future-proofing.
- **Error semantics**: typed error codes on the wire; client surfaces them.

### FlatBuffers wire schema (v1) — decisions (contract)

Defined in `schema/mooracer.fbs`, code-generated into the `wire` crate
(`mooracer-wire`) by `flatc --rust` in `wire/build.rs`; server and client share
that one generated module. `file_identifier "MOOR"`, `root_type Request`,
envelope version `1` (the `version` field on both `Request` and `Response`,
for future-proofing).

- **Value tree**: `table Value { kind: ValueKind; b: bool; i: long; f: double;
  s: string; arr: [Value]; keys: [string]; vals: [Value] }`. One table for all
  seven kinds, discriminated by `kind`; an object is `keys` + a parallel `vals`
  vector (insertion order preserved, mirroring the engine's ordered
  `Vec<(String, Value)>`). Only the field matching `kind` is meaningful.
- **Command union** (one payload per engine surface): `InsertCmd { docs }`,
  `FindCmd { filter, sort_field, sort_desc, skip, limit, one }`, `CountCmd`,
  `ExistsCmd`, `UpdateCmd { filter, update, many }`, `ReplaceCmd { filter,
  new_doc }`, `DeleteCmd { filter, many }`, `VectorSearchCmd`, `TextSearchCmd`,
  `HybridSearchCmd`, `GroupCmd` (full two-stage pipeline: query-level
  sort/skip/limit, then group, then group-level sort/limit, then the
  `AggFn`/`agg_field`), `StatsCmd`, `IndexCmd { kind, field, dim }` where
  `kind` is an `IndexKind` (`CreateValue`/`DropValue`/`CreateVector`/
  `DropVector`/`CreateText`/`DropText`). Index management is a write-path
  command (takes the exclusive lock) and is what enables search/value indexes
  at runtime from a client. Conventions inherited from the engine:
  `limit == 0` = no limit; `filter {}` = all; `one`/`many` flags collapse
  `find/find_one` and the `*_one`/`*_many` pairs onto one payload each.
- **Response union**: `InsertRes { ids }`, `FindRes { docs }`, `CountRes`,
  `ExistsRes`, `UpdateRes`, `ReplaceRes`, `DeleteRes { count }` (`delete_one`
  surfaces as `count ∈ {0,1}`), `SearchRes { hits: [SearchHit{doc, score}] }`
  (shared by all three search kinds), `GroupRes { groups }`, `StatsRes
  { docs, docs_memory, indexes, total_memory, per_index: [IndexStat] }`,
  `IndexRes` (success marker for `IndexCmd`).
- **Status codes** (`enum Status : byte`): `OK` plus the nine engine
  `StoreError` variants verbatim (`NotAnObject`, `IdMustBeString`,
  `DuplicateId`, `IdMismatch`, `NoIndex`, `PrimaryIndex`, `NoMatch`,
  `InvalidUpdate`, `VectorDimMismatch`), then four transport codes
  (`MalformedRequest`, `UnknownCommand`, `UnsupportedVersion`, `InternalError`).
  `Response.message` carries the human-readable detail (the engine's
  `Display` string); `Response.body` is present only on `OK`. The numeric
  discriminants of every enum/union are the wire format and are pinned by
  `wire/tests/schema.rs`.
- **Numeric widths**: counts / limits / ids / versions are `ulong` (u64 — no
  usize truncation across platforms), scores are `double`, vectors are
  `[float]` (f32, matching the vector index).

`wire/tests/schema.rs` round-trips every Value kind (lossless, order-preserving,
depth-nested), every command payload, every response body, every status code,
and runs the full FlatBuffers verifier over a request buffer.

## Client libraries
- **Rust client**: synchronous (and async via tokio if convenient) client that
  opens a connection and exposes the same Mongo-style chain API over the wire,
  returning native value trees.
- **Python client**: `mooracer` Python package speaking FlatBuffers over TCP,
  exposing the same chain API (`find_/ find_one / count / insert / update /
  delete / vector_search / text_search / hybrid_search / group`), returning
  native Python dicts. Pure Python (no Rust FFI) — it is a network client.
  *Rust client decisions* (fixed during implementation; authoritative for the
  Python client subtask — see `client-rust/src/lib.rs`):
  - Shapes: `Client::connect(addr) -> io::Result<Client>` owns one TCP
    connection + a **reused response buffer** + a wrapping `req_id` counter.
    `Client` is **not `Sync`** (one in-flight request at a time — the wire is
    a request/response loop); use one client per thread.
    `client.collection(name) -> Collection<'_>` (reborrowed `&mut`) scopes a
    named collection; all commands and the query chain live on it.
  - **Query chain** (`Collection::find(filter) -> Query`): lazy, mirroring the
    engine — `.sort(field, desc)` / `.skip(n)` / `.limit(m)` build; the single
    RPC runs only at a terminal `.to_list()` / `.first()` / `.count()` / `.find_one()`.
    `Query::group(field) -> GroupQuery` carries the query-level
    sort/skip/limit as the *pre-group* pipeline; `GroupQuery::sort` / `.limit`
    apply to the group documents and the terminal is `.agg(AggFn, field) ->
    Vec<Value>`. (There is **no** `Collection::group(filter)` — the group field
    comes from `find(filter).group(field)`, matching the engine.)
  - **Eager entry points** on `Collection`: `insert(doc) -> String` (the id),
    `insert_many(docs) -> Vec<String>` (ids in order), `find_one(filter) ->
    Option<Value>`, `count(filter) -> u64`, `exists(filter) -> bool`,
    `update_one`/`update_many(filter, update) -> u64`, `replace_one(filter,
    new_doc) -> u64`, `delete_one(filter) -> bool`, `delete_many(filter) ->
    u64`, `vector_search`/`text_search`/`hybrid_search(field, …, limit) ->
    Vec<(Value, f64)>` (score widened to f64), and `stats() -> Stats`.
  - **Typed errors**: `client::Error` = `Io(io::Error)` | `Protocol(String)` |
    `Server(Status, String)`. `Status` is the wire `Status` (the 9 engine
    `StoreError` codes + the 4 transport codes), so `update_one` with no match
    is `Error::Server(Status::NoMatch, …)` and a duplicate id is
    `Status::DuplicateId`; `Error::status()` returns the `Option<Status>`. A
    decode failure is `Protocol` (the connection is effectively poisoned).
  - **Synchronous only** for this build (the spec's "async via tokio if
    convenient" is deferred — an async wrapper would just wrap the blocking
    `Client` behind a spawn, so nothing is lost by leaving it for a later
    subtask).
- *Python client decisions* (fixed during implementation; see
  `client-python/mooracer/client.py`):
  - Package `client-python/mooracer/`: `client.py` (hand-written) + `wire/`
    (generated by `flatc --python` from the same `schema/mooracer.fbs`,
    checked in — regenerate with `flatc --python -o mooracer/
    ../schema/mooracer.fbs` and move `mooracer/mooracer/wire` → `mooracer/wire`).
  - API mirrors the Rust client: `Client.connect("host:port")` (one TCP
    connection, one in-flight request; one client per thread),
    `client.collection(name)`; lazy `find(filter) -> Query`
    (`.sort(field, desc)` / `.skip(n)` / `.limit(m)`; terminals
    `.to_list()` / `.first()` / `.find_one()` / `.count()`);
    `Query.group(field) -> GroupQuery` (query pipeline = pre-group stage;
    `.sort` / `.limit` = post-group; terminal `.agg(fn, field)` where `fn`
    is a name string or an `AggFn` number); eager
    `insert` / `insert_many` / `find_one` / `count` / `exists` /
    `update_one` / `update_many` / `replace_one` / `delete_one` (bool) /
    `delete_many` (int) / `vector_search` / `text_search` /
    `hybrid_search` (`[(dict, float)]`) / `stats` (dict).
  - **Native type mapping**: `dict`→Object (insertion order preserved),
    `list`/`tuple`→Array, `str`→Str, `int`→I64, `float`→F64, `bool`→Bool
    (checked BEFORE `int` — bool is an int subclass), `None`→Null. Non-str
    object keys and unencodable types raise `TypeError`.
  - **Typed errors** (mirror `client::Error`): `MooracerError` base;
    `MooracerIOError` (socket), `ProtocolError` (framing/decode failure —
    the connection is effectively poisoned), `ServerError(status, message)`
    with a `.name` property (the 9 engine codes + 4 transport codes, e.g.
    `NoMatch` = 7, `DuplicateId` = 3, `NoIndex` = 5).
  - The `"MOOR"` file identifier is written on every request (the server
    verifies it); `limit 0` = no limit; framing = 4-byte little-endian `u32`
    length prefix (same 256 MiB cap as the server); one reused receive
    buffer per connection.
  - **Indexes are managed over the wire** via `IndexCmd` (create/drop value,
    vector, and text indexes). The pytest suite can additionally drive a dev
    server (`server/src/bin/mooracer-devserver.rs`) that pre-creates
    vector/text indexes from env vars (`MOORACER_VECTOR_INDEX=coll:field:dim;…`,
    `MOORACER_TEXT_INDEX=coll:field;…`) before serving; the wire-level index
    commands are also covered directly. Tests insert docs over the wire and the
    engine maintains the indexes on insert.
    Unsorted finds return storage (hash-table) order — tests must compare
    sets unless a sort is applied.
  - Tests: `client-python/tests/` (pytest) — pure-protocol tests (value
    round-trips, request envelope, pinned enum discriminants) + live-TCP
    integration tests for every command family; run with
    `python3 -m pytest client-python/tests -q`.

## Benchmark harness (MooRacer-only report)
- A `bench` binary that runs representative workloads and reports, per workload:
  operations/second and latency p50/p99 (µs), with a configurable dataset size
  and thread/connection count. No external baseline required.
- Workloads at minimum: insert, insert_many, indexed find (equality), field
  range find, unindexed scan, sort+limit, update_one, delete, `$and`/`$or`
  filters, `$elemMatch`, vector_search, text_search (BM25), hybrid, aggregation
  group+agg.
- Produce a `BENCH.md` report (tables of ops/s + p50/p99 per workload) in the
  deliverable.

## Performance posture
- Optimize the hot paths (query evaluation, index lookup, serialization, network
  read/write) with allocation minimization, cache-friendly layouts, SIMD where it
  wins, and `unsafe` only where measured/profiled and locally documented.
- Provide `--release` builds (CPU-native where reasonable) for bench and server.

## Verification (quadruple method)
- **spec**: this document defines behavior + API.
- **code**: matches spec purpose and API.
- **tests**: a real `cargo test` suite covers every operator, search mode,
  aggregation, atomic batch, network round-trip, and both client libraries'
  happy + error paths. Each implementation subtask MUST come with Rust tests
  that pass; the Python client has Python unit tests (validated in-review; the
  loop's objective function is `cargo test`).
- **docs**: README (usage, API, wire protocol, config) + `BENCH.md` +
  `CHANGELOG.md` + the `examples/` directory.

## Notes for maintainers
- Keep the whole workspace compiling and green after every change
  (`cargo test` must pass; `cargo build --release` must not break).
- Where the spec is ambiguous, make a reasonable high-performance choice,
  record it in `CHANGELOG.md` (or the commit message), and optionally refine
  this spec.
- Do not add persistence, embedding, or any I/O the spec excludes.
