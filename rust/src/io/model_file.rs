//! Reading a model file as the C++ readers did through zstr::ifstream:
//! strict_fstream's open (an exception if the file cannot be opened; an
//! empty file opens, its peek only sets eof), then zstr's detection on
//! the first two bytes (gzip 1f 8b, zlib 78 01/9c/da: inflated; anything
//! else read as it is).

use std::io::Read;

extern "C" {
    fn strerror(errnum: i32) -> *const std::ffi::c_char;
}

/// The file's content, inflated if it is gzip or zlib data; Err is
/// strict_fstream's exception message when the file cannot be opened (a
/// read error ends the content, as the stream's would)
pub fn read(filename: &[u8]) -> Result<Vec<u8>, String> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::path::Path::new(std::ffi::OsStr::from_bytes(filename));
    let mut f = std::fs::File::open(path).map_err(|e| {
        // SAFETY: strerror returns a static NUL-terminated string
        let text = unsafe { std::ffi::CStr::from_ptr(strerror(e.raw_os_error().unwrap_or(0))) };
        format!(
            "strict_fstream: open('{}',in): open failed: {}",
            String::from_utf8_lossy(filename),
            text.to_string_lossy()
        )
    })?;
    let mut data = Vec::new();
    let _ = f.read_to_end(&mut data);
    let compressed = data.len() >= 2
        && ((data[0] == 0x1f && data[1] == 0x8b) || (data[0] == 0x78 && matches!(data[1], 0x01 | 0x9c | 0xda)));
    Ok(if compressed { inflate(&data) } else { data })
}

/// zstr's inflate (zlib's inflateInit2 with 15 + 32: a gzip or zlib
/// stream by its header), member after member to the end of the input.
/// A corrupt stream ends the program, as zstr's uncaught exception did
fn inflate(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = data;
    while !rest.is_empty() {
        let gzip = rest.len() >= 2 && rest[0] == 0x1f && rest[1] == 0x8b;
        let left = if gzip {
            let mut d = flate2::bufread::GzDecoder::new(rest);
            if let Err(e) = d.read_to_end(&mut out) {
                panic!("zstr::Exception: inflate: {e}");
            }
            d.into_inner().len()
        } else {
            let mut d = flate2::bufread::ZlibDecoder::new(rest);
            if let Err(e) = d.read_to_end(&mut out) {
                panic!("zstr::Exception: inflate: {e}");
            }
            d.into_inner().len()
        };
        if left == rest.len() {
            break;
        }
        rest = &rest[rest.len() - left..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn plain_gzip_and_zlib() {
        let dir = std::env::temp_dir().join(format!("crest-model-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let text = b"NAME test\nROWS\n N obj\nENDATA\n".repeat(100);
        let plain = dir.join("m.mps");
        std::fs::write(&plain, &text).unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&text).unwrap();
        let gz = gz.finish().unwrap();
        // Two gzip members, as zstr inflates them one after the other
        let gzf = dir.join("m.mps.gz");
        std::fs::write(&gzf, [gz.clone(), gz].concat()).unwrap();
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&text).unwrap();
        let zf = dir.join("m.z");
        std::fs::write(&zf, z.finish().unwrap()).unwrap();
        let empty = dir.join("e.mps");
        std::fs::write(&empty, b"").unwrap();
        let bytes = |p: &std::path::Path| p.to_str().unwrap().as_bytes().to_vec();
        assert_eq!(read(&bytes(&plain)).unwrap(), text);
        assert_eq!(read(&bytes(&gzf)).unwrap(), [text.clone(), text.clone()].concat());
        assert_eq!(read(&bytes(&zf)).unwrap(), text);
        assert_eq!(read(&bytes(&empty)).unwrap(), b"");
        assert!(read(&bytes(&dir.join("none"))).unwrap_err().ends_with("open failed: No such file or directory"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
