//! Text index — BM25 full-text search with Porter stemming over an inverted
//! index.
//!
//! Design notes (perf posture, see spec "Indexes" / "Search" /
//! "Performance posture"):
//!
//! - A [`TextIndex`] is created per **top-level field**. A document's field
//!   is a `Value::String` or a `Value::Array` of strings; anything else
//!   (including a missing field) is simply **not indexed** — a text index
//!   never rejects a write (unlike the vector index, which enforces a dim).
//! - **Tokenization** is one allocation-free pass: tokens are maximal runs
//!   of `[a-z0-9]` after lowercasing. **Porter stemming** (classic 1980
//!   algorithm) is applied to every token at write time and to query tokens
//!   at search time, so the postings table only ever holds stems.
//! - Inverted layout: `postings: HashMap<stem, Vec<(doc_idx, tf)>>` over
//!   parallel `ids` / `doc_lens` arrays. Posting lists stay sorted by doc
//!   index (insertions are monotonic). `doc_lens` + `total_tokens` give the
//!   BM25 length normalization with no per-query tokenization of the corpus.
//! - **BM25 (Okapi)**, Lucene's conventions: `k1 = 1.2`, `b = 0.75`,
//!   `idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))` (always positive),
//!   `score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`.
//!   Each distinct query stem is counted once (a bag-of-terms query).
//!   Documents with zero matching terms are not returned.
//! - Search walks only the posting lists of the query stems (inverted — no
//!   full corpus scan), accumulates scores into a scratch `Vec<f64>` sized
//!   to the corpus, then collects the positive entries and sorts them by
//!   (score descending, index order). `limit == 0` means no limit (same `0`
//!   convention as the query pipeline and the vector search).

use std::collections::HashMap;

use crate::value::Value;

/// BM25 term-frequency saturation (Lucene default).
pub const BM25_K1: f64 = 1.2;
/// BM25 document-length normalization (Lucene default).
pub const BM25_B: f64 = 0.75;

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/// True when `b` is an ASCII letter or digit.
#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// Extract the maximal `[a-z0-9]` runs of `s`, each lowercased
/// (one pass over the input; one exact-capacity `String` per token).
pub fn tokenize(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if is_word_byte(b[i]) {
            let start = i;
            while i < b.len() && is_word_byte(b[i]) {
                i += 1;
            }
            let mut tok = Vec::with_capacity(i - start);
            for &c in &b[start..i] {
                tok.push(if c.is_ascii_uppercase() {
                    c.to_ascii_lowercase()
                } else {
                    c
                });
            }
            // Lowercased ASCII-only bytes — no UTF-8 validation needed.
            out.push(unsafe { String::from_utf8_unchecked(tok) });
        } else {
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Porter stemmer (classic 1980 algorithm, ASCII)
// ---------------------------------------------------------------------------

#[inline]
fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Vowel test with the classic `y` rule: `y` is a vowel iff it is not the
/// first letter and is preceded by a consonant.
#[inline]
fn is_vowel_at(w: &[u8], i: usize) -> bool {
    let c = w[i];
    if is_vowel(c) {
        return true;
    }
    c == b'y' && i > 0 && !is_vowel_at(w, i - 1)
}

/// Measure `m` of `w[..len]`: the number of `(VC)` repetitions in the form
/// `[C](VC){m}[V]` — i.e. the number of **vowel→consonant transitions**
/// (a trailing consonant run counts once, a trailing vowel run does not;
/// `m("agree") = 1`, `m("fee") = 0`, `m("hop") = 1`, `m("plann") = 1`).
fn measure(w: &[u8], len: usize) -> usize {
    let mut m = 0;
    let mut in_vowel = false;
    for i in 0..len {
        if is_vowel_at(w, i) {
            in_vowel = true;
        } else if in_vowel {
            m += 1;
            in_vowel = false;
        }
    }
    m
}

/// Whether `w[..len]` contains a vowel.
#[inline]
fn has_vowel(w: &[u8], len: usize) -> bool {
    (0..len).any(|i| is_vowel_at(w, i))
}

/// Whether `w[k-1]`/`w[k]` are a double consonant (the `*d` condition,
/// e.g. "tt" and "ll"; the l/s/z non-halving guard is applied by callers).
#[inline]
fn doublec(w: &[u8], k: usize) -> bool {
    k >= 1 && w[k] == w[k - 1] && !is_vowel_at(w, k)
}

/// Whether `w[k-2..=k]` has the form consonant-vowel-consonant with a final
/// consonant that is not w, x, y (the `*o` condition, e.g. "hop" but not
/// "wag").
#[inline]
fn cvc(w: &[u8], k: usize) -> bool {
    k >= 2
        && !is_vowel_at(w, k)
        && is_vowel_at(w, k - 1)
        && !is_vowel_at(w, k - 2)
        && !matches!(w[k], b'w' | b'x' | b'y')
}

#[inline]
fn ends_with(w: &[u8], suffix: &str) -> bool {
    let s = suffix.as_bytes();
    w.len() >= s.len() && &w[w.len() - s.len()..] == s
}

/// Classic Porter stemmer (the 1980 published algorithm, verified line-by-
/// line against Martin Porter's reference C implementation). Input is a
/// single (lowercased) token — the engine only feeds it `tokenize` output.
pub fn porter_stem(token: &str) -> String {
    let mut w: Vec<u8> = token.as_bytes().to_vec();
    // Reference behavior: strings of length ≤ 2 are not stemmed.
    if w.len() > 2 {
        step_1ab(&mut w);
        if !w.is_empty() {
            step_1c(&mut w);
            step_2(&mut w);
            step_3(&mut w);
            step_4(&mut w);
            step_5(&mut w);
        }
    }
    String::from_utf8(w).expect("stem bytes are valid UTF-8")
}

fn step_1ab(w: &mut Vec<u8>) {
    // Step 1a.
    if w.last() == Some(&b's') {
        if ends_with(w, "sses") {
            w.truncate(w.len() - 2);
        } else if ends_with(w, "ies") {
            w.truncate(w.len() - 3);
            w.push(b'i');
        } else if w[w.len() - 2] != b's' {
            w.pop();
        }
    }
    // Step 1b. "eed" and "ed"/"ing" are mutually exclusive (if/else-if in
    // the reference): "feed" (m("f") = 0) is left untouched, "agreed"
    // loses its 'd' (EED → EE).
    if ends_with(w, "eed") {
        if measure(w, w.len() - 3) > 0 {
            w.pop();
        }
    } else if (ends_with(w, "ed") && has_vowel(w, w.len() - 2))
        || (ends_with(w, "ing") && has_vowel(w, w.len() - 3))
    {
        let strip = if ends_with(w, "ed") { 2 } else { 3 };
        w.truncate(w.len() - strip);
        let k = w.len() - 1;
        if ends_with(w, "at") || ends_with(w, "bl") || ends_with(w, "iz") {
            w.push(b'e');
        } else if doublec(w, k) {
            // Halve the double consonant, unless it is ll / ss / zz.
            let c = w.pop().unwrap();
            if matches!(w.last(), Some(&b'l') | Some(&b's') | Some(&b'z')) {
                w.push(c);
            }
        } else if measure(w, w.len()) == 1 && cvc(w, k) {
            w.push(b'e');
        }
    }
}

fn step_1c(w: &mut [u8]) {
    if w.last() == Some(&b'y') && has_vowel(w, w.len() - 1) {
        let i = w.len() - 1;
        w[i] = b'i';
    }
}

/// Apply a step-2/3 rule: if the whole word has m > 0 (checked by the
/// caller) AND the stem before `suffix` has m > 0, replace the suffix with
/// `rep`.
fn r_rule(w: &mut Vec<u8>, suffix: &str, rep: &str) -> bool {
    if !ends_with(w, suffix) || measure(w, w.len() - suffix.len()) == 0 {
        return false;
    }
    w.truncate(w.len() - suffix.len());
    w.extend_from_slice(rep.as_bytes());
    true
}

/// Step 2: double-suffix → single-suffix, keyed on the penultimate letter
/// (the reference's dispatch; longest suffix per key is tried first).
///
/// The rules are a dispatch where each condition is itself the side effect
/// (mutates `w` and reports whether it applied); clippy's `if_same_then_else`
/// flags the intentionally-empty branches.
#[allow(clippy::if_same_then_else)]
fn step_2(w: &mut Vec<u8>) {
    if w.len() < 2 || measure(w, w.len()) == 0 {
        return;
    }
    match w[w.len() - 2] {
        b'a' => {
            if r_rule(w, "ational", "ate") {
            } else if r_rule(w, "tional", "tion") {
            }
        }
        b'c' => {
            if r_rule(w, "enci", "ence") {
            } else if r_rule(w, "anci", "ance") {
            }
        }
        b'e' => {
            r_rule(w, "izer", "ize");
        }
        b'l' => {
            if r_rule(w, "abli", "able") {
            } else if r_rule(w, "alli", "al") {
            } else if r_rule(w, "entli", "ent") {
            } else if r_rule(w, "eli", "e") {
            } else {
                r_rule(w, "ousli", "ous");
            }
        }
        b'o' => {
            if r_rule(w, "ization", "ize") {
            } else if r_rule(w, "ation", "ate") {
            } else {
                r_rule(w, "ator", "ate");
            }
        }
        b's' => {
            if r_rule(w, "alism", "al") {
            } else if r_rule(w, "iveness", "ive") {
            } else if r_rule(w, "fulness", "ful") {
            } else {
                r_rule(w, "ousness", "ous");
            }
        }
        b't' => {
            if r_rule(w, "aliti", "al") {
            } else if r_rule(w, "iviti", "ive") {
            } else {
                r_rule(w, "biliti", "ble");
            }
        }
        _ => {}
    }
}

/// Step 3: -ic- / -ful / -ness family. Applied as the canonical **ordered
/// rule list** (Martin Porter 1980; byte-exact with the reference and with
/// NLTK's `ORIGINAL_ALGORITHM`): the first rule whose suffix matches AND whose
/// stem has `m > 0` wins. (A hand-rolled dispatch keyed on the wrong
/// penultimate character silently never fires — e.g. `hopeful`→`hope` needs
/// the `-ful` rule whose penultimate letter is `u`, not `l`; the ordered list
/// removes that whole class of bug.)
fn step_3(w: &mut Vec<u8>) {
    const RULES: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];
    for (suffix, rep) in RULES {
        if ends_with(w, suffix) && measure(w, w.len() - suffix.len()) > 0 {
            w.truncate(w.len() - suffix.len());
            w.extend_from_slice(rep.as_bytes());
            return;
        }
    }
}

/// Step 4: strip -ant/-ence/… when the stem's measure is > 1. `ion` only
/// after 's' or 't'; the 'o' key falls through to 'ism' (reference).
fn step_4(w: &mut Vec<u8>) {
    if w.len() < 2 {
        return;
    }
    let suffix = match w[w.len() - 2] {
        b'a' => ends_with(w, "al").then_some("al"),
        b'c' => {
            if ends_with(w, "ance") {
                Some("ance")
            } else {
                ends_with(w, "ence").then_some("ence")
            }
        }
        b'e' => ends_with(w, "er").then_some("er"),
        b'i' => ends_with(w, "ic").then_some("ic"),
        b'l' => {
            if ends_with(w, "able") {
                Some("able")
            } else {
                ends_with(w, "ible").then_some("ible")
            }
        }
        b'n' => {
            if ends_with(w, "ant") {
                Some("ant")
            } else if ends_with(w, "ement") {
                Some("ement")
            } else if ends_with(w, "ment") {
                Some("ment")
            } else {
                ends_with(w, "ent").then_some("ent")
            }
        }
        b'o' => {
            if ends_with(w, "ion") && w.len() > 4 && matches!(w[w.len() - 4], b's' | b't') {
                Some("ion")
            } else if ends_with(w, "ou") {
                Some("ou")
            } else {
                // fall-through from 'o' to the 's' key
                ends_with(w, "ism").then_some("ism")
            }
        }
        b's' => ends_with(w, "ism").then_some("ism"),
        b't' => {
            if ends_with(w, "ate") {
                Some("ate")
            } else {
                ends_with(w, "iti").then_some("iti")
            }
        }
        b'u' => ends_with(w, "ous").then_some("ous"),
        b'v' => ends_with(w, "ive").then_some("ive"),
        b'z' => ends_with(w, "ize").then_some("ize"),
        _ => None,
    };
    if let Some(suffix) = suffix
        && measure(w, w.len() - suffix.len()) > 1
    {
        w.truncate(w.len() - suffix.len());
    }
}

/// Step 5: drop a final 'e' when m > 1 (or m == 1 and not *CVC), and halve
/// a final 'll' when m > 1.
fn step_5(w: &mut Vec<u8>) {
    if w.last() == Some(&b'e') {
        let a = measure(w, w.len());
        let k = w.len() - 1;
        if a > 1 || (a == 1 && !cvc(w, k - 1)) {
            w.pop();
        }
    }
    if w.last() == Some(&b'l') && doublec(w, w.len() - 1) && measure(w, w.len()) > 1 {
        w.pop();
    }
}

// ---------------------------------------------------------------------------
// Field value → tokens
// ---------------------------------------------------------------------------

/// Coerce a document field value into its token stream (lowercased, Porter
/// stemmed). `None` when the field is missing or is not a string / array of
/// strings (such documents are simply not indexed — a text index never
/// rejects a write).
pub fn text_tokens(v: Option<&Value>) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    match v? {
        Value::Str(s) => {
            for t in tokenize(s) {
                out.push(porter_stem(&t));
            }
        }
        Value::Array(arr) => {
            for e in arr {
                let Value::Str(s) = e else {
                    return None;
                };
                for t in tokenize(s) {
                    out.push(porter_stem(&t));
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// The index
// ---------------------------------------------------------------------------

/// BM25 top-k result: the full document clone and its BM25 score (higher =
/// more relevant; only positive scores are returned).
pub type TextHit = (Value, f64);

/// A single-field BM25 text index (inverted, Porter-stemmed).
pub struct TextIndex {
    field: String,
    /// Document ids, in insertion order (index `i` owns doc `i`).
    ids: Vec<String>,
    /// Per-doc token counts (for BM25 length normalization).
    doc_lens: Vec<usize>,
    /// Per-doc distinct stems (keeps removal O(moved-doc terms): a
    /// `swap_remove` shifts one later doc left, and only *its* posting
    /// entries need their doc index rewritten).
    doc_terms: Vec<Vec<String>>,
    /// Corpus-wide token count (`sum(doc_lens)`).
    total_tokens: usize,
    /// Inverted table: stem → posting list `[(doc_idx, tf)]`, sorted by
    /// `doc_idx` (insertions are monotonic).
    postings: HashMap<String, Vec<(usize, u32)>>,
}

impl TextIndex {
    /// Create an empty index over `field`.
    pub fn new(field: &str) -> Self {
        TextIndex {
            field: field.to_string(),
            ids: Vec::new(),
            doc_lens: Vec::new(),
            doc_terms: Vec::new(),
            total_tokens: 0,
            postings: HashMap::new(),
        }
    }

    /// The indexed field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// `true` when no document is indexed.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The document ids currently indexed (insertion order).
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// The number of distinct indexed stems.
    pub fn num_terms(&self) -> usize {
        self.postings.len()
    }

    /// Index `id` with its token list (the caller has already tokenized and
    /// stemmed via [`text_tokens`]). Re-indexing an already-indexed id is a
    /// replace (the old entry is removed first).
    pub fn insert(&mut self, id: String, tokens: impl IntoIterator<Item = String>) {
        let tokens: Vec<String> = tokens.into_iter().collect();
        if let Some(i) = self.ids.iter().position(|s| s == &id) {
            self.remove_at(i);
        }
        let i = self.ids.len();
        // Per-term frequencies: tokens are short; sort + run-length is cheap
        // and allocation-light (no per-doc HashMap).
        let mut toks: Vec<&String> = tokens.iter().collect();
        toks.sort();
        let runs = run_lengths(&toks);
        let mut terms = Vec::with_capacity(runs.len());
        for (term, tf) in runs {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push((i, tf));
            terms.push(term.to_string());
        }
        self.ids.push(id);
        self.doc_lens.push(tokens.len());
        self.doc_terms.push(terms);
        self.total_tokens += tokens.len();
    }

    /// Remove the entry for `id` (a no-op when absent).
    pub fn remove(&mut self, id: &str) -> bool {
        match self.ids.iter().position(|s| s == id) {
            None => false,
            Some(i) => {
                self.remove_at(i);
                true
            }
        }
    }

    fn remove_at(&mut self, i: usize) {
        // 1) Drop every posting of doc i. Posting lists are sorted by doc
        //    idx, so a binary search finds i's slot in each of its terms.
        let mut emptied: Vec<String> = Vec::new();
        for term in &self.doc_terms[i] {
            let Some(list) = self.postings.get_mut(term) else {
                continue;
            };
            if let Ok(p) = list.binary_search_by_key(&i, |(d, _)| *d) {
                let (d, _) = list.remove(p);
                debug_assert_eq!(d, i);
                if list.is_empty() {
                    emptied.push(term.clone());
                }
            }
        }
        for term in emptied {
            self.postings.remove(&term);
        }
        self.total_tokens -= self.doc_lens[i];
        // 2) `swap_remove` shifts the last doc into slot i; rewrite *that
        //    doc's* entries (j → i) so every posting stays consistent.
        let last = self.ids.len() - 1;
        self.ids.swap_remove(i);
        self.doc_lens.swap_remove(i);
        self.doc_terms.swap_remove(i);
        if i != last {
            for term in &self.doc_terms[i] {
                let Some(list) = self.postings.get_mut(term) else {
                    continue;
                };
                if let Ok(p) = list.binary_search_by_key(&last, |(d, _)| *d) {
                    list[p].0 = i;
                }
            }
        }
    }

    /// Rebuild the whole index from `(id, tokens)` pairs (deterministic;
    /// used by `create_text_index` backfill).
    pub fn load(&mut self, pairs: Vec<(String, Vec<String>)>) {
        self.ids.clear();
        self.doc_lens.clear();
        self.doc_terms.clear();
        self.total_tokens = 0;
        self.postings.clear();
        for (id, tokens) in pairs {
            self.insert(id, tokens);
        }
    }

    /// BM25 search: the top `limit` documents by descending score
    /// (ties by index order), each as `(doc_idx, score)`. Only documents
    /// with a strictly positive score are returned. `limit == 0` means no
    /// limit. An empty index or an empty query (no tokens) returns empty.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f64)> {
        let n = self.ids.len();
        if n == 0 {
            return Vec::new();
        }
        // Query: tokenize, stem, dedupe (a bag-of-terms query).
        let mut qterms: Vec<String> = Vec::new();
        for t in tokenize(query) {
            qterms.push(porter_stem(&t));
        }
        qterms.sort();
        qterms.dedup();
        if qterms.is_empty() {
            return Vec::new();
        }

        let avgdl = self.total_tokens as f64 / n as f64;
        let mut score: Vec<f64> = vec![0.0; n];
        for term in &qterms {
            let Some(list) = self.postings.get(term) else {
                continue; // term absent from the corpus: contributes 0
            };
            let df = list.len() as f64;
            let idf = (1.0 + (n as f64 - df + 0.5) / (df + 0.5)).ln();
            for &(di, tf) in list {
                let f = tf as f64;
                let dl = self.doc_lens[di] as f64;
                score[di] += idf * (f * (BM25_K1 + 1.0))
                    / (f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl));
            }
        }

        let mut hits: Vec<(usize, f64)> = score
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > 0.0)
            .map(|(i, s)| (i, *s))
            .collect();
        // Best first; ties keep index order (stable sort + idx comparator).
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if limit > 0 {
            hits.truncate(limit);
        }
        hits
    }
}

/// Run-length frequencies of a sorted slice: `[(value, count)]`.
fn run_lengths<'a, T: PartialEq>(sorted: &'a [T]) -> Vec<(&'a T, u32)> {
    let mut out: Vec<(&'a T, u32)> = Vec::new();
    for t in sorted {
        match out.last_mut() {
            Some((last, count)) if **last == *t => *count += 1,
            _ => out.push((t, 1)),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_is_lowercase_alnum_runs() {
        assert_eq!(
            tokenize("The Quick, Brown Moo! 123-456."),
            vec!["the", "quick", "brown", "moo", "123", "456"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<String>>()
        );
        assert_eq!(tokenize("  "), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
        assert_eq!(
            tokenize("moo-cow_moo"),
            vec!["moo", "cow", "moo"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<String>>()
        );
        assert_eq!(
            tokenize("ÜBER-9"),
            vec!["ber".to_string(), "9".to_string()],
            "non-ASCII splits the run"
        );
    }

    #[test]
    fn porter_stemmer_classic_mappings() {
        let cases: &[(&str, &str)] = &[
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caressed", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("feeds", "feed"),
            // Classic Porter: step 5 strips the trailing 'e' from "agree"
            // (m("agre") == 1 and the *d/CVC condition on "agre" is false),
            // so both "agree" and "agreed" stem to "agre" — verified against
            // Martin Porter's reference algorithm (see AGENTS.md gotcha).
            ("agreed", "agre"),
            ("agree", "agre"),
            ("feeder", "feeder"),
            ("planned", "plan"),
            ("plan", "plan"),
            ("ceases", "ceas"),
            ("cease", "ceas"),
            ("controls", "control"),
            ("control", "control"),
            ("rails", "rail"),
            ("rail", "rail"),
            ("motorized", "motor"),
            ("sexualized", "sexual"),
            // Step 4 strips `-al` from "conditional"→"condition"→"condit"
            // (m("condit") == 2 > 1) and `-ion` from "condition"→"condit";
            // "consonant"→"conson", "matrices"→"matric" (verified against the
            // classic algorithm — see the AGENTS.md iteration-20 note).
            ("conditional", "condit"),
            ("condition", "condit"),
            ("consonant", "conson"),
            ("matrices", "matric"),
            ("ratified", "ratifi"),
            ("ratifies", "ratifi"),
            ("moo", "moo"),
            ("moos", "moo"),
            ("trials", "trial"),
            ("tried", "tri"),
            ("skied", "ski"),
            ("skies", "ski"),
            ("sky", "sky"),
            ("write", "write"),
            ("running", "run"),
            ("hopes", "hope"),
            ("hoping", "hope"),
            ("hop", "hop"),
            ("hopped", "hop"),
            ("hopping", "hop"),
            ("agrees", "agre"),
            ("hopeful", "hope"),
            ("planful", "plan"),
            ("formative", "form"),
            ("electrical", "electr"),
            ("rational", "ration"),
            ("relief", "relief"),
        ];
        for (input, expected) in cases {
            let got = porter_stem(input);
            assert_eq!(&got, expected, "stem({input})");
        }
    }

    #[test]
    fn text_tokens_coerces_strings_and_arrays() {
        let s = Value::str("The quick, brown moo (12x) MOOs!");
        assert_eq!(
            text_tokens(Some(&s)),
            Some(vec![
                "the".into(),
                "quick".into(),
                "brown".into(),
                "moo".into(),
                "12x".into(),
                "moo".into()
            ])
        );
        // array of strings, token streams concatenated in order
        let arr = Value::array_from(vec![Value::str("moo moo"), Value::str("the cow")]);
        assert_eq!(
            text_tokens(Some(&arr)),
            Some(vec!["moo".into(), "moo".into(), "the".into(), "cow".into()])
        );
        // missing / scalar / mixed array / empty string
        assert!(text_tokens(None).is_none());
        assert!(text_tokens(Some(&Value::i64(5))).is_none());
        assert!(
            text_tokens(Some(&Value::array_from(vec![
                Value::str("a"),
                Value::i64(1)
            ])))
            .is_none()
        );
        assert_eq!(text_tokens(Some(&Value::str(""))), Some(Vec::new()));
    }

    #[test]
    fn empty_index_searches_to_empty() {
        let ix = TextIndex::new("body");
        assert!(ix.is_empty());
        assert_eq!(ix.search("moo", 0), Vec::<(usize, f64)>::new());
    }

    #[test]
    fn term_frequency_and_rarity_boost_score() {
        let mut ix = TextIndex::new("body");
        // doc 0: "moo" appears 5x in a short doc; doc 1: "moo" 1x in a long
        // doc; doc 2: never mentions "moo".
        ix.insert("d0".into(), vec!["moo".to_string(); 5]);
        let mut d1 = vec!["moo".to_string()];
        d1.extend(std::iter::repeat_n("long".to_string(), 50));
        ix.insert("d1".into(), d1);
        ix.insert("d2".into(), vec!["other".to_string()]);
        assert_eq!(ix.len(), 3);
        assert_eq!(ix.num_terms(), 3);

        let res = ix.search("moo", 0);
        assert_eq!(res.len(), 2, "doc 2 has no match and is absent");
        assert_eq!(res[0].0, 0, "the 5x doc beats the 1x doc");
        assert_eq!(res[1].0, 1);
        assert!(res[0].1 > res[1].1);
        assert!(res[0].1 > 0.0);

        // "other" is just as rare as "moo" in its own doc (tf 1, short doc)
        // but rarer in the corpus (df 1 vs 2): its idf must win.
        let res2 = ix.search("other", 0);
        assert_eq!(res2.len(), 1);
        assert_eq!(res2[0].0, 2);
        let moo_tf1 = ix.search("moo", 0)[1].1;
        assert!(
            res2[0].1 > moo_tf1,
            "rarer term must outscore a common one at equal tf/length"
        );
    }

    #[test]
    fn stemming_at_query_time_matches() {
        let mut ix = TextIndex::new("body");
        ix.insert("a".into(), vec!["run".into()]); // e.g. from "runs"
        ix.insert("b".into(), vec!["ski".into()]); // e.g. from "skies"
        ix.insert("c".into(), vec!["moo".into()]);
        // "runs" and "run" both stem to "run".
        let res = ix.search("runs", 0);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0, "query 'runs' hits the 'run' doc");
        // "skied" and "skies" both stem to "ski" (the "ied" → "i" rule).
        let res = ix.search("skied", 0);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 1, "query 'skied' hits the 'ski' doc");
        // "moos" stems to "moo".
        let res = ix.search("moos", 0);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 2);
    }

    #[test]
    fn limit_is_top_k_and_zero_means_all() {
        let mut ix = TextIndex::new("body");
        for i in 0..10 {
            ix.insert(format!("d{i}"), vec!["moo".to_string(); i + 1]);
        }
        assert_eq!(ix.search("moo", 3).len(), 3, "limit 3 -> top 3");
        assert_eq!(ix.search("moo", 0).len(), 10, "limit 0 -> all");
        assert_eq!(ix.search("moo", 100).len(), 10, "limit > n -> all");
        // descending scores
        let res = ix.search("moo", 0);
        for w in res.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn query_tokens_not_in_corpus_match_nothing() {
        let mut ix = TextIndex::new("body");
        ix.insert("a".into(), vec!["moo".into()]);
        assert_eq!(ix.search("zebra", 0), Vec::<(usize, f64)>::new());
        assert_eq!(
            ix.search("!!  !!", 0),
            Vec::<(usize, f64)>::new(),
            "punctuation-only query has no tokens"
        );
    }

    #[test]
    fn remove_shrinks_and_updates_postings() {
        let mut ix = TextIndex::new("body");
        ix.insert("a".into(), vec!["moo".into(), "cow".into()]);
        ix.insert("b".into(), vec!["moo".into()]);
        assert!(ix.remove("a"));
        assert!(!ix.remove("a"), "second remove is a no-op");
        assert_eq!(ix.len(), 1);
        assert_eq!(ix.ids(), vec!["b"]);
        assert_eq!(ix.num_terms(), 1, "'cow' posting was dropped entirely");
        let res = ix.search("moo", 0);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        assert_eq!(ix.search("cow", 0), Vec::<(usize, f64)>::new());
    }

    #[test]
    fn reinsert_replaces() {
        let mut ix = TextIndex::new("body");
        ix.insert("a".into(), vec!["moo".into()]);
        ix.insert("a".into(), vec!["cow".into()]);
        assert_eq!(ix.len(), 1, "re-insert does not duplicate");
        assert_eq!(ix.ids(), vec!["a"]);
        assert_eq!(ix.num_terms(), 1, "the old stem posting is gone");
        assert_eq!(ix.search("moo", 0), Vec::<(usize, f64)>::new());
        assert_eq!(ix.search("cow", 0).len(), 1);
    }
}
