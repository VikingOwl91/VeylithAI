use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, ValueEnum)]
enum Sample {
    Original,
    Confirmation,
}

impl Sample {
    fn reference(self) -> Value {
        let report = match self {
            Self::Original => include_str!("../../v3/docs/routing-expert3-update500.json"),
            Self::Confirmation => {
                include_str!("../../v3/docs/routing-expert3-update500-confirmation.json")
            }
        };
        serde_json::from_str(report).expect("invalid bundled expert 3 report")
    }
}

#[derive(Parser)]
#[command(version, about = "Reproduce the frozen Veylith expert 3 study on CPU")]
struct Args {
    #[arg(long, value_enum, default_value = "original")]
    sample: Sample,
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    /// New report path. Its parent directory must exist.
    #[arg(long)]
    out: PathBuf,
    /// V3 source directory, relative to the current working directory.
    #[arg(long, default_value = "veylith/v3")]
    v3_dir: PathBuf,
    #[arg(long, default_value = "python3")]
    python: PathBuf,
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn verify_file(path: &Path, expected: &Value) -> Result<()> {
    let mut input = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    io::copy(&mut input, &mut hash)?;
    let actual = format!("{:x}", hash.finalize());
    if Some(actual.as_str()) != expected.as_str() {
        return Err(format!("SHA-256 mismatch: {}", path.display()).into());
    }
    Ok(())
}

fn verify_inputs(args: &Args, reference: &Value) -> Result<()> {
    let expected = &reference["sha256"];
    let corpus = args.data_root.join("corpus");
    let manifest = corpus.join("manifest.json");
    verify_file(&manifest, &expected["manifest"])?;
    let metadata = read_json(&manifest)?;
    let tokenizer = metadata["tokenization"]["tokenizer"]["model"]
        .as_str()
        .ok_or("missing tokenizer model in the corpus manifest")?;
    verify_file(&corpus.join(tokenizer), &expected["tokenizer"])?;
    verify_file(&args.checkpoint, &expected["checkpoint"])?;
    for (path, digest) in reference["source_sha256"]
        .as_object()
        .ok_or("missing evaluator hashes")?
    {
        verify_file(&args.v3_dir.join(path), digest)?;
    }
    Ok(())
}

fn verify_report(report: &Value, reference: &Value) -> Result<()> {
    // Git state, elapsed time and installed package versions describe the new run.
    for field in [
        "experiment",
        "schema_version",
        "update",
        "layer",
        "split",
        "primary",
        "delta",
        "bootstrap",
        "sample_batch_offset_per_source",
        "config",
        "sha256",
        "source_sha256",
        "sequences",
        "substitute_counts",
        "outcomes",
        "unaffected_prefix_ce_exact",
        "baseline_reference_exact",
    ] {
        if report.get(field) != reference.get(field) {
            return Err(format!("report differs from frozen reference at {field}").into());
        }
    }
    for key in ["device", "dtype", "threads", "interop_threads"] {
        if report["runtime"].get(key) != reference["runtime"].get(key) {
            return Err(format!("report differs from frozen reference at runtime.{key}").into());
        }
    }
    Ok(())
}

fn run(args: Args, started: Instant) -> Result<()> {
    if args.out.try_exists()? {
        return Err("output already exists; choose a new report path".into());
    }
    let parent = args
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let scratch = tempfile::Builder::new()
        .prefix(".veylith-")
        .tempdir_in(parent)?;
    let staged = scratch.path().join("report.json");
    let reference = args.sample.reference();
    verify_inputs(&args, &reference)?;
    let inputs_verified = Instant::now();
    let mut worker = Command::new("nice");
    worker
        .arg("-n")
        .arg("15")
        .arg("--")
        .arg(&args.python)
        .arg("-B")
        .arg(args.v3_dir.join("analysis/expert3_study.py"))
        .arg("--data-root")
        .arg(&args.data_root)
        .arg("--checkpoint")
        .arg(&args.checkpoint)
        .args(["--threads", "2"])
        .arg("--out")
        .arg(&staged)
        .env("OMP_NUM_THREADS", "2")
        .env("MKL_NUM_THREADS", "2")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("HIP_VISIBLE_DEVICES", "")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::from(io::stderr()));
    if matches!(args.sample, Sample::Confirmation) {
        worker.arg("--fresh-sample");
    }
    let status = worker.status()?;
    if !status.success() {
        return Err(format!("expert 3 evaluator exited with {status}").into());
    }
    let evaluated = Instant::now();
    verify_report(&read_json(&staged)?, &reference)?;
    // Same filesystem, atomic publication, and no clobber if another run won the race.
    fs::hard_link(&staged, &args.out)?;
    let published = Instant::now();
    println!(
        "{}",
        json!({
            "experiment": reference["experiment"],
            "sample": match args.sample { Sample::Original => "original", Sample::Confirmation => "confirmation" },
            "verified": true,
            "preflight_seconds": (inputs_verified - started).as_secs_f64(),
            "evaluator_seconds": (evaluated - inputs_verified).as_secs_f64(),
            "verification_and_publish_seconds": (published - evaluated).as_secs_f64(),
            "total_seconds": (published - started).as_secs_f64(),
        })
    );
    Ok(())
}

fn main() -> ExitCode {
    let started = Instant::now();
    match run(Args::parse(), started) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("veylith-driver: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_the_selected_sample_and_each_outcome() {
        let original = Sample::Original.reference();
        let second = Sample::Confirmation.reference();
        assert!(verify_report(&second, &original).is_err());
        let mut result = original.clone();
        result["git_commit"] = json!("different checkout");
        result["elapsed_seconds"] = json!(0.5);
        verify_report(&result, &original).unwrap();
        result["outcomes"]["replaced"]["delta_ce"] = json!(0.0);
        assert!(verify_report(&result, &original).is_err());
    }
}
