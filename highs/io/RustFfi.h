/**@file io/RustFfi.h
 * @brief The repr(C) types shared by the file readers ported to Rust
 * (rust/src/io)
 */
#ifndef IO_RUST_FFI_H_
#define IO_RUST_FFI_H_

#include <string>
#include <vector>

// A borrowed Rust array
template <typename T>
struct RsSlice {
  const T* ptr;
  size_t len;
  std::vector<T> vec() const { return std::vector<T>(ptr, ptr + len); }
  std::string str() const { return std::string(ptr, len); }
};

struct RsMessage {
  int kind;  // HighsLogType, 0 for highsLogDev, or -1 for printf
  RsSlice<char> text;
};

#endif
