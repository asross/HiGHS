//! The crest binary links the C++ that is not ported yet: libhighs.a of a
//! static HIGHS_RUST CMake build (-DHIGHS_RUST=ON -DBUILD_SHARED_LIBS=OFF),
//! in HIGHS_LIB_DIR, with the C++ standard library and, when that build
//! found it, zlib. The library and its tests need nothing.
use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=HIGHS_LIB_DIR");
    if env::var_os("CARGO_FEATURE_CREST").is_none() {
        return;
    }
    let dir = env::var("HIGHS_LIB_DIR")
        .expect("crest: set HIGHS_LIB_DIR to the lib directory of a static HIGHS_RUST build");
    let lib = Path::new(&dir).join("libhighs.a");
    assert!(lib.exists(), "crest: {} not found", lib.display());
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rustc-link-arg-bin=crest={}", lib.display());
    let hconfig = std::fs::read_to_string(Path::new(&dir).join("../HConfig.h")).unwrap_or_default();
    if hconfig.contains("#define ZLIB_FOUND") {
        println!("cargo:rustc-link-arg-bin=crest=-lz");
    }
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-arg-bin=crest=-lc++");
    } else {
        for l in ["-lstdc++", "-lpthread", "-ldl"] {
            println!("cargo:rustc-link-arg-bin=crest={l}");
        }
    }
}
