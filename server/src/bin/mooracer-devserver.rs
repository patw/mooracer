//! mooracer-devserver — test-support server for the Python client tests.
//!
//! Identical protocol behavior to `mooracer-server`; the only difference is
//! that it can pre-create **indexes** on collections from the environment
//! before serving (the v1 wire protocol has no index-management command, and
//! the Python test process cannot otherwise reach into the server's store).
//! Docs are NOT seeded here — the tests insert them over the wire, and the
//! indexes are maintained on insert (the engine's normal maintenance path).
//!
//! Environment:
//!   MOORACER_ADDR           "host:port" (default 127.0.0.1:4141)
//!   MOORACER_VECTOR_INDEX   "coll:field:dim;coll2:field2:dim2"
//!   MOORACER_TEXT_INDEX     "coll:field;coll2:field2"

use mooracer_engine::Collection;
use mooracer_server::Server;
use std::collections::HashMap;
use std::sync::RwLock;

fn main() -> std::io::Result<()> {
    let addr =
        std::env::var("MOORACER_ADDR").unwrap_or_else(|_| "127.0.0.1:4141".to_string());

    let server = Server::new();
    let listener = std::net::TcpListener::bind(&addr)?;
    let addr = listener.local_addr()?;

    let state: &RwLock<HashMap<String, Collection>> = server.state();
    let mut write = state.write().unwrap();

    if let Ok(spec) = std::env::var("MOORACER_VECTOR_INDEX") {
        for entry in spec.split(';').filter(|s| !s.is_empty()) {
            let mut parts = entry.split(':');
            let (coll, field, dim) = match (parts.next(), parts.next(), parts.next()) {
                (Some(c), Some(f), Some(d)) => (c.to_string(), f.to_string(), d),
                _ => panic!("bad MOORACER_VECTOR_INDEX entry {entry:?} (want coll:field:dim)"),
            };
            let dim: usize = dim.parse().expect("dim must be an integer");
            write.entry(coll.clone()).or_insert_with(|| Collection::new(coll)).create_vector_index(&field, dim);
        }
    }
    if let Ok(spec) = std::env::var("MOORACER_TEXT_INDEX") {
        for entry in spec.split(';').filter(|s| !s.is_empty()) {
            let mut parts = entry.split(':');
            let (coll, field) = match (parts.next(), parts.next()) {
                (Some(c), Some(f)) => (c.to_string(), f.to_string()),
                _ => panic!("bad MOORACER_TEXT_INDEX entry {entry:?} (want coll:field)"),
            };
            write.entry(coll.clone()).or_insert_with(|| Collection::new(coll)).create_text_index(&field);
        }
    }
    drop(write);

    println!("mooracer-devserver listening on {addr}");
    server.run(&listener)
}
