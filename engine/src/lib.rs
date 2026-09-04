//! mooracer-engine — core in-memory document data engine.
//!
//! So far: the native [`Value`] value tree (compact enum, path-based
//! get/set, Mongo-compatible numeric semantics, JSON-like `Display`), the
//! [`Collection`] document store (insert / insert_many, `_id` rules), the
//! index layer (always-present primary `_id` index, ordered field indexes
//! for equality + range, maintenance primitives for delete/update), and the
//! lazy [`Query`] builder (Mongo-style `find` / `find_one` / `count` /
//! `exists` with the `.sort(field, descending)` / `.skip(n)` / `.limit(n)`
//! pipeline and `.to_list()` / `.first()` / `.count()` terminals), and the
//! write API (update operators `$set`/`$inc`/`$unset`, `replace_one`,
//! `delete_one`/`delete_many`, and an atomic batch [`Transaction`] via
//! `Collection::begin`), the search layer (vector cosine, BM25 text, RRF
//! hybrid), and aggregation (`.find(…).group(field).agg(fn, field)` with
//! count / sum / mean / min / max / collect / first / last and optional
//! group sort/limit).

pub mod agg;
pub mod collection;
pub mod index;
pub mod query;
pub mod text;
pub mod value;
pub mod vector;

pub use agg::{AggFn, GroupQuery};
pub use collection::{
    Collection, CollectionStats, HybridHit, IndexStats, RRF_K, StoreError, Transaction,
};
pub use index::{FieldIndex, IndexSet};
pub use query::Query;
pub use text::{TextHit, TextIndex};
pub use value::{PathError, Value};
pub use vector::{VectorHit, VectorIndex};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_crate_scaffolds() {
        assert_eq!(VERSION, "0.1.0");
        assert!(std::env!("CARGO_PKG_NAME").starts_with("mooracer"));
    }

    #[test]
    fn value_is_exported() {
        let v: Value = "moo".into();
        assert_eq!(v.type_name(), "str");
        assert_eq!(v.as_str(), Some("moo"));
    }
}
