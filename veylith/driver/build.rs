use std::process::Command;

fn output(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned())
}

fn main() {
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
