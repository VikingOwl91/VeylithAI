mod inputs;
mod model;
mod stats;

use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
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

    fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Confirmation => "confirmation",
        }
    }
}

#[derive(Parser)]
#[command(
    version,
    about = "Evaluate and verify the frozen expert 3 study in Rust on CPU"
)]
struct Args {
    #[arg(long, value_enum, default_value = "original")]
    sample: Sample,
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    /// New report path. Its parent directory must exist and have room for checkpoint normalization.
    #[arg(long)]
    out: PathBuf,
}

fn verified_hash(path: &Path, expected: &Value) -> Result<String> {
    let mut input = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    io::copy(&mut input, &mut hash)?;
    let actual = format!("{:x}", hash.finalize());
    if Some(actual.as_str()) != expected.as_str() {
        return Err(format!("SHA-256 mismatch: {}", path.display()).into());
    }
    Ok(actual)
}

fn verify_inputs(args: &Args, reference: &Value) -> Result<Value> {
    let expected = &reference["sha256"];
    let corpus = args.data_root.join("corpus");
    let manifest = corpus.join("manifest.json");
    let manifest_hash = verified_hash(&manifest, &expected["manifest"])?;
    let metadata: Value = serde_json::from_reader(BufReader::new(File::open(manifest)?))?;
    let tokenizer = metadata["tokenization"]["tokenizer"]["model"]
        .as_str()
        .ok_or("missing tokenizer model in the corpus manifest")?;
    Ok(json!({
        "manifest": manifest_hash,
        "tokenizer": verified_hash(&corpus.join(tokenizer), &expected["tokenizer"])?,
        "checkpoint": verified_hash(&args.checkpoint, &expected["checkpoint"])?,
    }))
}

fn source_hashes() -> Value {
    let sources: &[(&str, &[u8])] = &[
        ("driver/Cargo.toml", include_bytes!("../Cargo.toml")),
        ("driver/Cargo.lock", include_bytes!("../Cargo.lock")),
        ("driver/build.rs", include_bytes!("../build.rs")),
        ("driver/src/main.rs", include_bytes!("main.rs")),
        ("driver/src/inputs.rs", include_bytes!("inputs.rs")),
        ("driver/src/model.rs", include_bytes!("model.rs")),
        ("driver/src/stats.rs", include_bytes!("stats.rs")),
    ];
    let mut hashes = serde_json::Map::new();
    for (path, bytes) in sources {
        hashes.insert(
            (*path).into(),
            json!(format!("{:x}", Sha256::digest(bytes))),
        );
    }
    Value::Object(hashes)
}

fn verify_report(report: &Value, reference: &Value) -> Result<()> {
    for field in [
        "experiment",
        "update",
        "layer",
        "split",
        "primary",
        "delta",
        "bootstrap",
        "sample_batch_offset_per_source",
        "config",
        "sha256",
        "sequences",
        "substitute_counts",
        "unaffected_prefix_ce_exact",
        "baseline_reference_exact",
    ] {
        if report.get(field) != reference.get(field) {
            return Err(format!("report differs from frozen reference at {field}").into());
        }
    }
    if report["outcomes"] != reference["outcomes"] {
        for (name, outcome) in reference["outcomes"].as_object().expect("bundled outcomes") {
            for (field, expected) in outcome.as_object().expect("bundled outcome") {
                let actual = &report["outcomes"][name][field];
                if actual != expected {
                    return Err(format!(
                        "outcomes.{name}.{field}: expected {expected}, measured {actual}"
                    )
                    .into());
                }
            }
        }
        return Err("unexpected outcome fields".into());
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
    // Linux priority calls with ID zero apply to the calling thread. Set it before
    // LibTorch starts its worker pool so newly created threads inherit the priority.
    unsafe {
        let current = libc::getpriority(libc::PRIO_PROCESS, 0);
        if current < 15 && libc::setpriority(libc::PRIO_PROCESS, 0, 15) != 0 {
            return Err(io::Error::last_os_error().into());
        }
    }
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    let _no_grad = tch::no_grad_guard();
    let reference = args.sample.reference();
    let mut hashes = verify_inputs(&args, &reference)?;
    let inputs_verified = Instant::now();
    let sample_hash = reference["sha256"]["sample_tokens"]
        .as_str()
        .expect("bundled sample digest");
    let batches = inputs::load_batches(
        &args.data_root,
        matches!(args.sample, Sample::Confirmation),
        sample_hash,
    )?;
    hashes["sample_tokens"] = json!(sample_hash); // Ordered bytes were checked by load_batches.
    let model = model::Model::new(inputs::load_weights(&args.checkpoint, scratch.path())?)?;
    scratch.close()?;
    let loaded = Instant::now();
    let mut stats = stats::Stats::new();
    for (index, (tokens, punctuation)) in batches.iter().enumerate() {
        let batch = model.evaluate_batch(tokens, punctuation)?;
        stats.add_batch(
            &batch.ordinary,
            &batch.altered,
            punctuation,
            &batch.eligible,
            &batch.substitute,
        )?;
        let number = index + 1;
        if number == 1 || number % 5 == 0 {
            eprintln!("Batch {number}/{} complete", batches.len());
        }
    }
    let inferred = Instant::now();
    let mut report = stats.finish()?;
    for name in ["all", "sentence_end"] {
        for field in ["baseline_ce", "targets"] {
            if report["outcomes"][name][field] != reference["outcomes"][name][field] {
                return Err(
                    format!("ordinary {name} {field} differs from the frozen reference").into(),
                );
            }
        }
    }
    let evaluated = Instant::now();
    report["baseline_reference_exact"] = json!(true);
    // Frozen study parameters are shared. All outcome fields come from the current run.
    for field in [
        "experiment",
        "config",
        "update",
        "layer",
        "split",
        "primary",
        "delta",
        "bootstrap",
        "sample_batch_offset_per_source",
    ] {
        report[field] = reference[field].clone();
    }
    report["schema_version"] = json!(2);
    report["git_commit"] = json!(option_env!("DRIVER_BUILD_COMMIT"));
    report["git_dirty"] = json!(option_env!("DRIVER_BUILD_DIRTY").map(|v| v == "true"));
    report["git_state_scope"] = json!("driver package at build time");
    report["source_sha256"] = source_hashes();
    report["sha256"] = hashes;
    report["runtime"] = json!({
        "evaluator": "rust", "rustc": option_env!("DRIVER_RUSTC"), "tch": "0.27.0",
        "libtorch_target": "2.15.0", "device": "cpu", "dtype": "float32",
        "threads": tch::get_num_threads(), "interop_threads": tch::get_num_interop_threads(),
    });
    report["reproduction_command"] = json!(format!(
        "veylith-driver --sample {} --data-root \"$DATA\" --checkpoint \"$CKPT\" --out \"$OUT\"",
        args.sample.name()
    ));
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    verify_report(&report, &reference)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.persist_noclobber(&args.out)?;
    let published = Instant::now();
    println!(
        "{}",
        json!({
            "experiment": reference["experiment"], "sample": args.sample.name(), "verified": true,
            "preflight_seconds": (inputs_verified - started).as_secs_f64(),
            "evaluator_seconds": (evaluated - inputs_verified).as_secs_f64(),
            "load_seconds": (loaded - inputs_verified).as_secs_f64(),
            "inference_and_reduction_seconds": (inferred - loaded).as_secs_f64(),
            "bootstrap_seconds": (evaluated - inferred).as_secs_f64(),
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
        result["source_sha256"] = source_hashes();
        verify_report(&result, &original).unwrap();
        result["outcomes"]["replaced"]["delta_ce"] = json!(0.0);
        assert!(verify_report(&result, &original).is_err());
    }
}
