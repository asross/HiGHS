//! crest: the highs app's command line (rust/src/lp_data/app.rs), linked
//! with the C++ that is not ported yet (libhighs.a of a static HIGHS_RUST
//! build, see build.rs). The `Highs` instance and the loaded options are
//! C++ (highs/lp_data/HighsAppRust.cpp).
use highs_rs::lp_data::app::{app_main, AppHost};
use std::os::unix::ffi::OsStrExt;

extern "C" {
    fn highs_app_create(host: *mut AppHost) -> *mut std::ffi::c_void;
    fn highs_app_destroy(ctx: *mut std::ffi::c_void);
}

fn main() {
    let args: Vec<Vec<u8>> = std::env::args_os().map(|a| a.as_bytes().to_vec()).collect();
    let mut host = std::mem::MaybeUninit::<AppHost>::uninit();
    // SAFETY: highs_app_create fills every field of host and returns the
    // context it holds, freed once the app returns
    let status = unsafe {
        let ctx = highs_app_create(host.as_mut_ptr());
        let status = app_main(&args, &*host.as_ptr());
        highs_app_destroy(ctx);
        status
    };
    std::process::exit(status);
}
