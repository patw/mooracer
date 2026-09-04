//! mooracer-server — binary entry point.
//!
//! Binds the address from `MOORACER_ADDR` (default `127.0.0.1:4141`) and the
//! pool size from `MOORACER_THREADS` (default 8), then serves until the
//! listener errors. All protocol logic lives in the `mooracer_server` library.

use mooracer_server::{DEFAULT_POOL_SIZE, Server};

fn main() -> std::io::Result<()> {
    let addr = std::env::var("MOORACER_ADDR").unwrap_or_else(|_| "127.0.0.1:4141".to_string());
    let pool_size: usize = std::env::var("MOORACER_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_POOL_SIZE);

    let server = Server::with_pool_size(pool_size);
    let listener = std::net::TcpListener::bind(&addr)?;
    println!(
        "mooracer-server listening on {addr} (pool={})",
        server.pool_size()
    );
    server.run(&listener)
}
