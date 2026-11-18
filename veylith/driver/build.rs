use std::process::Command;

fn output(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned())
}

fn main() {
    #[cfg(feature = "rocm")]
    {
        let root = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let torch = std::env::var("LIBTORCH").unwrap_or_else(|_| "/usr".into());
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=LIBTORCH");
        cc::Build::new()
            .cpp(true)
            .std("c++20")
            .define("__HIP_PLATFORM_AMD__", None)
            .include(format!("{root}/include"))
            .include(format!("{torch}/include"))
            .file("src/rocm.cpp")
            .compile("veylith_hip_stream");
        println!("cargo:rustc-link-search=native={root}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_hip,--as-needed");
    }
    for path in ["src", "build.rs", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let mut refs = vec![
        "HEAD".to_owned(),
        "index".to_owned(),
        "packed-refs".to_owned(),
    ];
    if let Some(branch) = output("git", &["symbolic-ref", "-q", "HEAD"]) {
        refs.push(branch);
    }
    for name in refs {
        if let Some(path) = output("git", &["rev-parse", "--git-path", &name])
            && std::path::Path::new(&path).exists()
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(commit) = output("git", &["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=DRIVER_BUILD_COMMIT={commit}");
    }
    if let Some(status) = output("git", &["status", "--porcelain", "--", "."]) {
        println!("cargo:rustc-env=DRIVER_BUILD_DIRTY={}", !status.is_empty());
    }
    let compiler = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    if let Some(version) = output(&compiler, &["--version"]) {
        println!("cargo:rustc-env=DRIVER_RUSTC={version}");
    }
}
