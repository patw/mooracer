//! Codegen build step for the MooRacer wire protocol.
//!
//! Runs `flatc --rust -o <OUT_DIR>` over `schema/mooracer.fbs` (workspace
//! root) and emits `mooracer_generated.rs` into `OUT_DIR`, which
//! `src/lib.rs` includes. (`-o` rather than `--out`: older flatc builds,
//! incl. 23.5.x, only accept `-o`.)
//!
//! Requirements: the `flatc` compiler (FlatBuffers) on `PATH`, or at the
//! path named by the `FLATC` environment variable. See the "flatc usage"
//! section of AGENTS.md. `flatc` 23.5.x generated code is verified to
//! compile against the `flatbuffers` 25.x crate (see AGENTS.md, iteration 22).

use std::process::Command;

fn main() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let schema = manifest_dir.join("../schema/mooracer.fbs");
    if !schema.exists() {
        panic!("mooracer-wire: schema not found at {}", schema.display());
    }
    println!("cargo:rerun-if-changed={}", schema.display());

    let out_dir = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
    let flatc = std::env::var("FLATC").unwrap_or_else(|_| "flatc".to_string());

    let status = Command::new(&flatc)
        .args(["--rust", "-o", &out_dir])
        .arg(&schema)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "mooracer-wire: cannot run `{flatc}` ({e}). \
                 Install the FlatBuffers compiler (flatc) or set FLATC=/path/to/flatc. \
                 See AGENTS.md 'flatc usage' for details."
            )
        });
    if !status.success() {
        panic!("mooracer-wire: `flatc --rust` failed with {status}");
    }

    // Sanity: the generated file exists (flatc names it <schema>_generated.rs).
    let generated = std::path::Path::new(&out_dir).join("mooracer_generated.rs");
    if !generated.exists() {
        panic!(
            "mooracer-wire: flatc did not produce {} (schema file name must be mooracer.fbs)",
            generated.display()
        );
    }
}
