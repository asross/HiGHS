//! crest: the highs app's command line (rust/src/lp_data/app.rs) on the
//! Rust Highs object (rust/src/lp_data/highs.rs); all Rust.
use highs_rs::lp_data::app::app_main;
use highs_rs::lp_data::lp_handle::highs::{app_create, app_destroy};
use std::os::unix::ffi::OsStrExt;

fn main() {
    let args: Vec<Vec<u8>> = std::env::args_os().map(|a| a.as_bytes().to_vec()).collect();
    let (host, ctx) = app_create();
    let status = app_main(&args, &host);
    // SAFETY: the context of app_create, freed once
    unsafe { app_destroy(ctx) };
    std::process::exit(status);
}
