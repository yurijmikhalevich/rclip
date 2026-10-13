use std::path::Path;
use std::process::Command;

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=src/imaging/raw_params.cpp");
  build_libraw();
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
    link_compiler_runtime();
  }
}

/// Builds the vendored LibRaw with lossy (JPEG) and deflate DNG support, like rawpy's build.
fn build_libraw() {
  let root = Path::new("vendor/libraw");
  let mut build = cc::Build::new();
  build.cpp(true).include(root).warnings(false).extra_warnings(false).define("LIBRAW_NODLL", None);
  build.define("USE_JPEG", None).define("USE_JPEG8", None).define("USE_ZLIB", None);
  for variable in ["DEP_TURBOJPEG_INCLUDE", "DEP_Z_INCLUDE"] {
    for include in std::env::var(variable).unwrap_or_default().split(',').filter(|path| !path.is_empty()) {
      build.include(include);
    }
  }
  if build.get_compiler().is_like_msvc() {
    build.define("WIN32", None).define("_CRT_SECURE_NO_WARNINGS", None);
  } else {
    build.flag_if_supported("-pthread");
  }
  let mut sources = vec![root.join("src/libraw_c_api.cpp"), root.join("src/libraw_datastream.cpp")];
  for entry in std::fs::read_dir(root.join("src")).expect("the LibRaw sources are vendored") {
    let directory = entry.expect("the LibRaw sources are readable").path();
    if directory.is_dir() {
      let mut files: Vec<_> = std::fs::read_dir(&directory)
        .expect("the LibRaw sources are readable")
        .map(|entry| entry.expect("the LibRaw sources are readable").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cpp"))
        .collect();
      files.sort();
      sources.extend(files);
    }
  }
  sources.push("src/imaging/raw_params.cpp".into());
  build.files(&sources).compile("raw");
}

/// The prebuilt ONNX Runtime library checks the OS version with `@available`, which compiles to
/// calls into clang's compiler runtime. Rust links with `-nodefaultlibs`, so link it explicitly.
fn link_compiler_runtime() {
  let output = Command::new("clang").arg("--print-resource-dir").output().expect("clang must be installed");
  let resource_dir = String::from_utf8(output.stdout).expect("the clang resource dir is UTF-8");
  println!("cargo:rustc-link-search=native={}/lib/darwin", resource_dir.trim());
  println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
