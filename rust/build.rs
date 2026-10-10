//! The git hash of the logs' header and --version (`git describe
//! --always`, as CMake's HIGHS_GITHASH; "n/a" outside a git checkout).
//! Nothing is linked: crest and the library are all Rust.
use std::process::Command;

fn main() {
    let hash = Command::new("git")
        .args(["describe", "--always"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "n/a".into());
    println!("cargo:rustc-env=CREST_GITHASH={hash}");
}
