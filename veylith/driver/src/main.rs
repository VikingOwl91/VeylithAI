mod inputs;
mod matched;
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
    fn reference_text(self) -> &'static str {
        match self {
            Self::Original => include_str!("../../v3/docs/routing-expert3-update500.json"),
            Self::Confirmation => {
                include_str!("../../v3/docs/routing-expert3-update500-confirmation.json")
            }
        }
    }

    fn reference(self) -> Value {
        serde_json::from_str(self.reference_text()).expect("invalid bundled expert 3 report")
    }

    fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Confirmation => "confirmation",
        }
    }
}

/// Float metrics are in nats. Counts, hashes and same-device controls remain exact.
const GPU_CE_ATOL: f64 = 1e-4;

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Study {
    /// Replace expert 3 at every eligible punctuation piece.
    Expert3,
    /// Compare separate, single-position substitutions on matched punctuation and words.
    Matched,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Backend {
    Cpu,
    Rocm,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Rocm => "rocm:0",
        }
    }

    fn start(self) -> Result<(tch::Device, Option<i32>)> {
        if self == Self::Cpu {
            return Ok((tch::Device::Cpu, None));
        }
        #[cfg(feature = "rocm")]
        {
            unsafe extern "C" {
                fn veylith_start_rocm(
                    priority: *mut i32,
                    error: *mut libc::c_char,
                    len: usize,
                ) -> i32;
            }
            let mut priority = 0;
            let mut error = [0; 1024];
            // The helper handles C++ exceptions and NUL-terminates any error.
            if unsafe { veylith_start_rocm(&mut priority, error.as_mut_ptr(), error.len()) } != 0 {
                return Err(unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
                    .into());
            }
            Ok((tch::Device::Cuda(0), Some(priority)))
        }
        #[cfg(not(feature = "rocm"))]
        Err("build with --features rocm to run on an AMD GPU".into())
    }
}

#[derive(Parser)]
#[command(
    version,
    about = "Evaluate and verify the frozen expert 3 study in Rust"
)]
struct Args {
    #[arg(long, value_enum, default_value = "expert3")]
    study: Study,
    /// Publish the matched study's eligibility and balance without evaluating losses.
    #[arg(long)]
    coverage_only: bool,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
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
        ("veylith/driver/Cargo.toml", include_bytes!("../Cargo.toml")),
        ("Cargo.toml", include_bytes!("../../../Cargo.toml")),
        ("Cargo.lock", include_bytes!("../../../Cargo.lock")),
        ("veylith/driver/build.rs", include_bytes!("../build.rs")),
        ("veylith/driver/src/main.rs", include_bytes!("main.rs")),
        ("veylith/driver/src/inputs.rs", include_bytes!("inputs.rs")),
        (
            "veylith/driver/src/matched.rs",
            include_bytes!("matched.rs"),
        ),
        ("veylith/driver/src/model.rs", include_bytes!("model.rs")),
        #[cfg(feature = "rocm")]
        ("veylith/driver/src/rocm.cpp", include_bytes!("rocm.cpp")),
        ("veylith/driver/src/stats.rs", include_bytes!("stats.rs")),
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

fn verify_report(report: &Value, reference: &Value, backend: Backend) -> Result<()> {
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
    ] {
        if report.get(field) != reference.get(field) {
            return Err(format!("report differs from frozen reference at {field}").into());
        }
    }
    let mut checked = report["outcomes"].clone();
    if backend == Backend::Rocm {
        for name in reference["outcomes"].as_object().unwrap().keys() {
            let actual = &mut checked[name];
            let expected = &reference["outcomes"][name];
            for field in ["baseline_ce", "delta_ce", "sequence_bootstrap_ci95"] {
                let compare = |v: &mut Value, target: &Value| -> Result<()> {
                    let measurement = v.as_f64().ok_or("missing measured CE")?;
                    let error =
                        (measurement - target.as_f64().ok_or("missing reference CE")?).abs();
                    if error > GPU_CE_ATOL {
                        return Err(format!("outcomes.{name}.{field}: measured {measurement}, expected {target}, absolute difference {error} exceeds {GPU_CE_ATOL}").into());
                    }
                    *v = target.clone();
                    Ok(())
                };
                if field == "sequence_bootstrap_ci95" {
                    let interval = actual[field]
                        .as_array_mut()
                        .ok_or("missing measured interval")?;
                    if interval.len() != 2 {
                        return Err("expected two interval endpoints".into());
                    }
                    for (v, target) in interval.iter_mut().zip(expected[field].as_array().unwrap())
                    {
                        compare(v, target)?;
                    }
                } else {
                    compare(&mut actual[field], &expected[field])?;
                }
            }
        }
    }
    if checked != reference["outcomes"] {
        return Err("outcome fields differ from frozen reference".into());
    }
    let exact = ["all", "sentence_end"].iter().all(|name| {
        report["outcomes"][name]["baseline_ce"] == reference["outcomes"][name]["baseline_ce"]
    });
    if report["baseline_reference_exact"] != json!(exact) || (backend == Backend::Cpu && !exact) {
        return Err("inconsistent baseline exactness".into());
    }
    if report["runtime"]["device"] != backend.name() {
        return Err("incorrect runtime device".into());
    }
    for key in ["dtype", "threads", "interop_threads"] {
        if report["runtime"].get(key) != reference["runtime"].get(key) {
            return Err(format!("report differs from frozen reference at runtime.{key}").into());
        }
    }
    Ok(())
}

fn expert3_report(
    model: &model::Model,
    batches: &[inputs::Batch],
    reference: &Value,
) -> Result<(Value, Instant)> {
    let mut stats = stats::Stats::new();
    for (index, batch) in batches.iter().enumerate() {
        let result = model.evaluate_batch(&batch.tokens, &batch.punctuation)?;
        stats.add_batch(
            &result.ordinary,
            &result.altered,
            &batch.punctuation,
            &result.eligible,
            &result.substitute,
        )?;
        let number = index + 1;
        if number == 1 || number % 5 == 0 {
            eprintln!("Batch {number}/{} complete", batches.len());
        }
    }
    let inferred = Instant::now();
    let mut report = stats.finish()?;
    report["baseline_reference_exact"] = json!(
        ["all", "sentence_end"]
            .iter()
            .all(|name| report["outcomes"][name]["baseline_ce"]
                == reference["outcomes"][name]["baseline_ce"])
    );
    for field in ["experiment", "primary", "delta", "bootstrap"] {
        report[field] = reference[field].clone();
    }
    Ok((report, inferred))
}

fn matched_report(
    model: &model::Model,
    batches: &[inputs::Batch],
    reference: &Value,
    coverage_only: bool,
) -> Result<(Value, Instant)> {
    let mut selections = Vec::with_capacity(batches.len());
    for batch in batches {
        let (ids, confidence) = model.ordinary_routes(&batch.tokens)?;
        let selection = matched::select_pairs(&batch.punctuation, &batch.words, &ids, &confidence)?;
        selections.push(selection);
    }
    let adequate = matched::coverage_adequate(&selections)?;
    let mut coverage = matched::coverage_report(&selections)?;
    eprintln!(
        "Pre-outcome matched coverage by source: {}",
        coverage["by_source"]
    );
    let (mut report, inferred) = if !adequate || coverage_only {
        coverage["baseline_all_targets"] = json!(0);
        coverage["baseline_all_ce"] = Value::Null;
        coverage["matched_effects"] = Value::Null;
        coverage["unaffected_prefix_ce_exact"] = Value::Null;
        (coverage, Instant::now())
    } else {
        let mut stats = matched::MatchedStats::new();
        for (index, (batch, selection)) in batches.iter().zip(&selections).enumerate() {
            let loss = model.evaluate_matched(
                &batch.tokens,
                &selection.punctuation_mask,
                &selection.word_mask,
            )?;
            stats.add_batch(selection, &loss.ordinary, &loss.punctuation, &loss.word)?;
            let number = index + 1;
            if number == 1 || number % 5 == 0 {
                eprintln!("Matched batch {number}/{} complete", batches.len());
            }
        }
        let inferred = Instant::now();
        (stats.finish()?, inferred)
    };
    if report["support_adequate"] != json!(adequate) {
        return Err("inconsistent matched cohort support".into());
    }
    let scored = adequate && !coverage_only;
    if scored
        && (report["baseline_all_ce"] != reference["outcomes"]["all"]["baseline_ce"]
            || report["baseline_all_targets"] != reference["outcomes"]["all"]["targets"]
            || report["unaffected_prefix_ce_exact"] != true
            || report["matched_effects"].is_null())
    {
        return Err(
            "matched study failed the frozen all-target baseline, prefix or effect check".into(),
        );
    }
    if !scored
        && (!report["matched_effects"].is_null()
            || !report["baseline_all_ce"].is_null()
            || !report["unaffected_prefix_ce_exact"].is_null())
    {
        return Err("unexpected loss measurements in a coverage-only report".into());
    }
    report["baseline_reference_exact"] = if scored { json!(true) } else { Value::Null };
    report["scored"] = json!(scored);
    report["stage"] = json!(if scored {
        "matched_losses"
    } else if !adequate {
        "insufficient_support"
    } else {
        "coverage_only"
    });
    report["experiment"] = json!("v31-expert3-matched-punctuation-word");
    report["primary"] =
        json!("punctuation minus matched word in substitution-induced next-token CE change");
    Ok((report, inferred))
}

fn run(args: Args, started: Instant) -> Result<()> {
    if args.study == Study::Matched && args.device != Backend::Cpu {
        return Err("the matched study requires --device cpu".into());
    }
    if args.coverage_only && args.study != Study::Matched {
        return Err("--coverage-only requires --study matched".into());
    }
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
    let (device, priority) = args.device.start()?;
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
    let model = model::Model::new(
        inputs::load_weights(&args.checkpoint, scratch.path())?,
        device,
    )?;
    scratch.close()?;
    let loaded = Instant::now();
    let (mut report, inferred) = match args.study {
        Study::Expert3 => expert3_report(&model, &batches, &reference)?,
        Study::Matched => matched_report(&model, &batches, &reference, args.coverage_only)?,
    };
    let evaluated = Instant::now();
    // Both studies use the same frozen model, validation corpus and sample selection.
    for field in [
        "config",
        "update",
        "layer",
        "split",
        "sample_batch_offset_per_source",
    ] {
        report[field] = reference[field].clone();
    }
    report["schema_version"] = json!(2);
    report["git_commit"] = json!(option_env!("DRIVER_BUILD_COMMIT"));
    report["git_dirty"] = json!(option_env!("DRIVER_BUILD_DIRTY").map(|v| v == "true"));
    report["git_state_scope"] =
        json!("driver package and workspace manifest and lockfile at build time");
    report["source_sha256_path_base"] = json!("repository root");
    report["source_sha256"] = source_hashes();
    report["sha256"] = hashes;
    if args.study == Study::Matched {
        report["sample"] = json!(args.sample.name());
        report["baseline_report_sha256"] = json!(format!(
            "{:x}",
            Sha256::digest(args.sample.reference_text().as_bytes())
        ));
    }
    report["runtime"] = json!({
        "evaluator": "rust", "rustc": option_env!("DRIVER_RUSTC"), "tch": "0.27.0",
        "libtorch_target": "2.15.0", "device": args.device.name(), "dtype": "float32",
        "threads": tch::get_num_threads(), "interop_threads": tch::get_num_interop_threads(),
    });
    if let Some(priority) = priority {
        report["runtime"]["hip_stream_priority"] = json!(priority);
        report["runtime"]["roc_global_cu_mask"] = json!(std::env::var("ROC_GLOBAL_CU_MASK").ok());
        report["verification_ce_absolute_tolerance"] = json!(GPU_CE_ATOL);
    }
    let mode = match (args.study, args.coverage_only) {
        (Study::Expert3, _) => "",
        (Study::Matched, false) => " --study matched",
        (Study::Matched, true) => " --study matched --coverage-only",
    };
    report["reproduction_command"] = json!(format!(
        "veylith-driver{mode} --device {} --sample {} --data-root \"$DATA\" --checkpoint \"$CKPT\" --out \"$OUT\"",
        if args.device == Backend::Rocm {
            "rocm"
        } else {
            "cpu"
        },
        args.sample.name()
    ));
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    if args.study == Study::Expert3
        && let Err(err) = verify_report(&report, &reference, args.device)
    {
        if args.device == Backend::Rocm {
            eprintln!("Unverified GPU report: {report}");
        }
        return Err(err);
    }
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.persist_noclobber(&args.out)?;
    let published = Instant::now();
    println!(
        "{}",
        json!({
            "experiment": report["experiment"], "sample": args.sample.name(), "verified": true,
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
    fn matched_options_reject_invalid_combinations_before_loading_inputs() {
        let run_with = |options: &[&str]| {
            let args = Args::parse_from(
                [
                    "veylith-driver",
                    "--data-root",
                    "unused",
                    "--checkpoint",
                    "unused",
                    "--out",
                    "unused",
                ]
                .into_iter()
                .chain(options.iter().copied()),
            );
            run(args, Instant::now()).unwrap_err().to_string()
        };
        assert_eq!(
            run_with(&["--study", "matched", "--device", "rocm"]),
            "the matched study requires --device cpu"
        );
        assert_eq!(
            run_with(&["--coverage-only"]),
            "--coverage-only requires --study matched"
        );
    }

    #[test]
    fn checks_the_selected_sample_and_each_outcome() {
        let original = Sample::Original.reference();
        let second = Sample::Confirmation.reference();
        assert!(verify_report(&second, &original, Backend::Cpu).is_err());
        let mut result = original.clone();
        result["source_sha256"] = source_hashes();
        verify_report(&result, &original, Backend::Cpu).unwrap();
        result["outcomes"]["replaced"]["delta_ce"] = json!(0.0);
        assert!(verify_report(&result, &original, Backend::Cpu).is_err());
    }
}
