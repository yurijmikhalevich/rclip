use std::process::Command;

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
    link_compiler_runtime();
  }
}

/// The prebuilt ONNX Runtime library checks the OS version with `@available`, which compiles to
/// calls into clang's compiler runtime. Rust links with `-nodefaultlibs`, so link it explicitly.
fn link_compiler_runtime() {
  let output = Command::new("clang").arg("--print-resource-dir").output().expect("clang must be installed");
  let resource_dir = String::from_utf8(output.stdout).expect("the clang resource dir is UTF-8");
  println!("cargo:rustc-link-search=native={}/lib/darwin", resource_dir.trim());
  println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
