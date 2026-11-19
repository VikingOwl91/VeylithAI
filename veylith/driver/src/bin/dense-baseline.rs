//! Fixed-token VEYLITH-8 causal dense baseline against v3.1-L.
#[path = "../dense_data.rs"]
mod dense_data;
#[path = "../dense_model.rs"]
mod dense_model;

use clap::{Parser, ValueEnum};
use dense_data::{Corpus, SOURCES};
use dense_model::{Adam, CONTEXT, Dense, HEADS, UPDATES, VOCAB, WIDTH};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
use tch::{Device, Kind, Reduction, Tensor};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TARGETS: usize = 64 * 511;
const VAL_COUNTS: [i64; 3] = [310531, 1296602, 511097];

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum Backend {
    Cpu,
    Rocm,
}

#[derive(Parser)]
#[command(about = "Train or resume the fixed VEYLITH-8 plain dense Transformer")]
struct Args {
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    /// Divide each logical 8-row batch for gradient accumulation and validation.
    #[arg(long, default_value_t = 4)]
    microbatch: i64,
    #[arg(long, default_value_t = 25)]
    save_every: usize,
    /// Stop after a completed update and write a resumable partial checkpoint.
    #[arg(long, default_value_t=UPDATES)]
    stop_after: usize,
    #[arg(long)]
    resume: bool,
}

fn device(backend: Backend) -> Result<(Device, Option<i32>)> {
    if backend == Backend::Cpu {
        return Ok((Device::Cpu, None));
    }
    #[cfg(not(feature = "rocm"))]
    return Err("ROCm requires the rocm build feature".into());
    #[cfg(feature = "rocm")]
    {
        unsafe extern "C" {
            fn veylith_start_rocm(priority: *mut i32, error: *mut libc::c_char, len: usize) -> i32;
        }
        let mut error = [0; 1024];
        let mut priority = 0;
        if unsafe { veylith_start_rocm(&mut priority, error.as_mut_ptr(), error.len()) } != 0 {
            return Err(unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned()
                .into());
        }
        Ok((Device::Cuda(0), Some(priority)))
    }
}

const MIN_FREE: usize = 6 << 30;
const ALLOCATION_SLACK: usize = 2 << 30;

fn loaded_libraries(rocm: bool) -> Result<BTreeMap<String, String>> {
    let maps = fs::read_to_string("/proc/self/maps")?;
    let mut output = BTreeMap::new();
    for line in maps.lines() {
        let Some(index) = line.find('/') else {
            continue;
        };
        let path = Path::new(&line[index..]);
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let cpu_lib = ["libtorch_cpu.so", "libc10.so", "libtorch.so"]
            .iter()
            .any(|s| name.starts_with(s));
        let gpu_lib = ["libamdhip64.so", "libtorch_hip.so", "libc10_hip.so"]
            .iter()
            .any(|s| name.starts_with(s));
        if !(cpu_lib || rocm && gpu_lib) || output.contains_key(name) {
            continue;
        }
        let id = Command::new("readelf")
            .env("LC_ALL", "C")
            .args(["-nW"])
            .arg(path)
            .output()
            .ok()
            .filter(|v| v.status.success())
            .and_then(|v| {
                let text = String::from_utf8(v.stdout).ok()?;
                let (_, tail) = text.split_once("Build ID: ")?;
                let id = tail.split_whitespace().next()?;
                (id.len() >= 16 && id.bytes().all(|c| c.is_ascii_hexdigit()))
                    .then(|| format!("elf:{id}"))
            });
        let fingerprint = match id {
            Some(id) => id,
            None => {
                let mut sha = Sha256::new();
                io::copy(&mut File::open(path)?, &mut sha)?;
                format!("sha256:{:x}", sha.finalize())
            }
        };
        output.insert(name.to_owned(), fingerprint);
    }
    for required in ["libtorch_cpu.so", "libc10.so"]
        .into_iter()
        .chain(rocm.then_some("libamdhip64.so"))
        .chain(rocm.then_some("libtorch_hip.so"))
    {
        if !output.keys().any(|k| k.starts_with(required)) {
            return Err(format!("missing loaded {required} identity").into());
        }
    }
    Ok(output)
}

#[cfg(feature = "rocm")]
unsafe extern "C" {
    fn veylith_dense_rocm_identity(
        identity: *mut libc::c_char,
        capacity: usize,
        err: *mut libc::c_char,
        length: usize,
    ) -> i32;
    fn veylith_dense_rocm_memory(
        free: *mut usize,
        total: *mut usize,
        err: *mut libc::c_char,
        length: usize,
    ) -> i32;
}

fn backend_identity(backend: Backend) -> Result<Value> {
    let libraries = loaded_libraries(backend == Backend::Rocm)?;
    if backend == Backend::Cpu {
        return Ok(json!({"libraries":libraries}));
    }
    #[cfg(not(feature = "rocm"))]
    return Err("ROCm unavailable".into());
    #[cfg(feature = "rocm")]
    {
        let mut identity = [0; 512];
        let mut error = [0; 1024];
        if unsafe {
            veylith_dense_rocm_identity(
                identity.as_mut_ptr(),
                identity.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        } != 0
        {
            return Err(unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned()
                .into());
        }
        Ok(json!({"libraries":libraries, "runtime_and_device":unsafe {
            std::ffi::CStr::from_ptr(identity.as_ptr()) }.to_str()?}))
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct MemoryRange {
    min_free: Option<usize>,
    max_free: Option<usize>,
    total_bytes: Option<usize>,
    samples: u64,
}
impl MemoryRange {
    fn record(&mut self, free: usize, total: usize) {
        self.min_free = Some(self.min_free.map_or(free, |m| m.min(free)));
        self.max_free = Some(self.max_free.map_or(free, |m| m.max(free)));
        self.total_bytes = Some(total);
        self.samples += 1;
    }
    fn include(&mut self, old: &Self) {
        if let (Some(low), Some(high)) = (old.min_free, old.max_free) {
            self.min_free = Some(self.min_free.map_or(low, |m| m.min(low)));
            self.max_free = Some(self.max_free.map_or(high, |m| m.max(high)));
            self.samples += old.samples;
        }
    }
}

struct MemoryGuard<'a> {
    backend: Backend,
    dir: &'a Path,
    range: MemoryRange,
    durable_step: Option<usize>,
}
impl MemoryGuard<'_> {
    fn check(&mut self, phase: &str, before_allocation: bool) -> Result<()> {
        if self.backend == Backend::Cpu {
            return Ok(());
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (phase, before_allocation);
            Err("ROCm unavailable".into())
        }
        #[cfg(feature = "rocm")]
        {
            let mut free = 0usize;
            let mut total = 0usize;
            let mut error = [0; 1024];
            let rc = unsafe {
                veylith_dense_rocm_memory(&mut free, &mut total, error.as_mut_ptr(), error.len())
            };
            let required = MIN_FREE
                + if before_allocation {
                    ALLOCATION_SLACK
                } else {
                    0
                };
            if rc == 0 {
                self.range.record(free, total);
                if free >= required {
                    return Ok(());
                }
            }
            let probe_error = (rc != 0).then(|| {
                unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            });
            let event = json!({"status":"guard_stopped","phase":phase,"free_bytes":(rc==0).then_some(free),
                "required_bytes":required,"floor_bytes":MIN_FREE,"probe_error":probe_error,
                "last_durable_update":self.durable_step,"memory":self.range});
            let mut staged = tempfile::NamedTempFile::new_in(self.dir)?;
            serde_json::to_writer_pretty(&mut staged, &event)?;
            writeln!(staged)?;
            staged.as_file().sync_all()?;
            staged.persist(self.dir.join("vram-stop.json"))?;
            File::open(self.dir)?.sync_all()?;
            Err(format!("VRAM guard stopped at {phase}; see vram-stop.json").into())
        }
    }
}

fn token_losses(model: &Dense, ids: &Tensor) -> Tensor {
    let logits = model.logits(&ids.narrow(1, 0, CONTEXT - 1));
    let labels = ids.narrow(1, 1, CONTEXT - 1).reshape([-1]);
    logits
        .flatten(0, 1)
        .cross_entropy_loss(&labels, None::<&Tensor>, Reduction::None, -100, 0.)
        .reshape([-1, CONTEXT - 1])
}

fn validate(model: &Dense, data: &Corpus, micro: i64, guard: &mut MemoryGuard) -> Result<Value> {
    let _guard = tch::no_grad_guard();
    let mut sources = BTreeMap::new();
    let mut macro_ce = 0.;
    let mut total_nats = 0.;
    let mut total_targets = 0;
    let start = Instant::now();
    for (index, name) in SOURCES.iter().enumerate() {
        let mut targets = 0i64;
        let mut nats = 0.;
        for batch in data.validation_batches(index) {
            let (ids, mask) = batch?;
            for (part, valid) in ids.split(micro, 0).into_iter().zip(mask.split(micro, 0)) {
                guard.check("before validation microbatch", true)?;
                targets += valid.sum(Kind::Int64).int64_value(&[]);
                let ce = token_losses(model, &part.to_device(model.device()));
                nats += ce
                    .masked_select(&valid.to_device(model.device()))
                    .sum(Kind::Double)
                    .double_value(&[]);
                guard.check("after validation microbatch", false)?;
            }
        }
        if targets != VAL_COUNTS[index] || !nats.is_finite() {
            return Err(format!(
                "invalid validation measurement for {name}: {targets} targets, {nats} nats"
            )
            .into());
        }
        let ce = nats / targets as f64;
        sources.insert(*name, json!({"main_ce":ce,"targets":targets,"nats":nats}));
        macro_ce += ce / SOURCES.len() as f64;
        total_targets += targets;
        total_nats += nats;
    }
    Ok(
        json!({"macro_main_ce":macro_ce,"token_main_ce":total_nats/total_targets as f64,
              "targets":total_targets, "sources":sources,"seconds":start.elapsed().as_secs_f64()}),
    )
}

#[derive(Serialize, Deserialize)]
struct State {
    protocol: Value,
    step: usize,
    consumed_targets: usize,
    initial_validation: Value,
    train_seconds: f64,
    active_wall_seconds: f64,
    train_curve: Vec<Value>,
    memory: MemoryRange,
}

fn checkpoint(dir: &Path, opt: &Adam, state: &State) -> Result<()> {
    let tmp = tempfile::Builder::new()
        .prefix("dense-")
        .suffix(".ot")
        .tempfile_in(dir)?;
    opt.save(tmp.path(), &serde_json::to_vec(state)?)?;
    tmp.as_file().sync_all()?;
    tmp.persist(dir.join("latest.ot"))?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn train_update(
    model: &Dense,
    opt: &mut Adam,
    data: &Corpus,
    micro: i64,
    guard: &mut MemoryGuard,
) -> Result<(f64, f64)> {
    let mut sum = Tensor::zeros([], (Kind::Float, model.device()));
    for logical in 0..8 {
        let batch = data.train_batch(opt.step, logical)?;
        if batch.size() != [8, CONTEXT] {
            return Err("expected eight full training rows".into());
        }
        for part in batch.split(micro, 0) {
            guard.check("before training microbatch", true)?;
            let loss = token_losses(model, &part.to_device(model.device())).sum(Kind::Float)
                / TARGETS as f64;
            sum += loss.detach();
            loss.backward();
            guard.check("after training microbatch", false)?;
        }
    }
    let mean = sum.double_value(&[]);
    if !mean.is_finite() {
        return Err("nonfinite train loss".into());
    }
    guard.check("before optimizer update", true)?;
    let norm = opt.update()?;
    if let Device::Cuda(gpu) = model.device() {
        tch::Cuda::synchronize(gpu as i64);
    }
    guard.check("after optimizer update", false)?;
    Ok((mean, norm))
}

fn protocol(args: &Args, data: &Corpus, priority: Option<i32>, identity: Value) -> Value {
    let src: [(&str, &[u8]); 8] = [
        ("cli", include_bytes!("dense-baseline.rs")),
        ("model", include_bytes!("../dense_model.rs")),
        ("data", include_bytes!("../dense_data.rs")),
        ("rocm_bridge", include_bytes!("../rocm.cpp")),
        (
            "workspace_manifest",
            include_bytes!("../../../../Cargo.toml"),
        ),
        ("workspace_lock", include_bytes!("../../../../Cargo.lock")),
        ("crate_manifest", include_bytes!("../../Cargo.toml")),
        ("crate_build", include_bytes!("../../build.rs")),
    ];
    let hashes: BTreeMap<_, _> = src
        .iter()
        .map(|(name, b)| (*name, format!("{:x}", Sha256::digest(b))))
        .collect();
    json!({
        "schema":2, "experiment":"VEYLITH-8", "architecture":"pre-LN, learned positions, full causal MHA, GELU MLP, untied LM head",
        "layers":1, "width":WIDTH, "heads":HEADS, "ff_width":4*WIDTH, "vocab":VOCAB,
        "dropout":0, "dtype":"fp32", "seed":0, "block":CONTEXT, "updates":UPDATES,
        "effective_batch_sequences":64,"microbatch_sequences":args.microbatch,"target_budget":UPDATES*TARGETS,
        "adamw":{"betas":[0.9,0.999],"eps":1e-8,"weight_decay_matrices":0.01,"clip_norm":1.0,
            "base_lr":0.0003,"warmup":15,"min_lr_ratio":0.1},
        "device":format!("{:?}",args.device),"hip_stream_priority":priority,
        "cpu_threads":2,"interop_threads":1,
        "rocm_cu_mask":std::env::var("ROC_GLOBAL_CU_MASK").ok(),
        "hsa_cu_mask":std::env::var("HSA_CU_MASK").ok(),
        "source_sha256":hashes, "rocm_feature":cfg!(feature="rocm"),"backend_identity":identity,
        "vram_floor_bytes":(args.device==Backend::Rocm).then_some(MIN_FREE),
        "vram_transient_slack_bytes":(args.device==Backend::Rocm).then_some(ALLOCATION_SLACK),
        "corpus":data.evidence()
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    if ![1, 2, 4, 8].contains(&args.microbatch)
        || args.save_every == 0
        || !(1..=UPDATES).contains(&args.stop_after)
    {
        return Err("invalid microbatch/save interval/stop step".into());
    }
    let start = Instant::now();
    unsafe {
        if libc::getpriority(libc::PRIO_PROCESS, 0) < 15
            && libc::setpriority(libc::PRIO_PROCESS, 0, 15) != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    fs::create_dir_all(&args.run_dir)?;
    let lock = File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .open(args.run_dir.join("run.lock"))?;
    lock.try_lock()
        .map_err(|e| format!("run directory locked: {e}"))?;
    let latest = args.run_dir.join("latest.ot");
    let report = args.run_dir.join("result.json");
    if report.exists() || latest.exists() != args.resume {
        return Err(
            "completed report exists, or --resume does not match available checkpoint".into(),
        );
    }
    if args.device == Backend::Rocm
        && (std::env::var("ROC_GLOBAL_CU_MASK").as_deref() != Ok("0xff")
            || std::env::var("ROCR_VISIBLE_DEVICES").as_deref() != Ok("0"))
    {
        return Err(
            "dense ROCm training requires ROCR_VISIBLE_DEVICES=0 and ROC_GLOBAL_CU_MASK=0xff"
                .into(),
        );
    }
    let (device, priority) = device(args.device)?;
    let identity = backend_identity(args.device)?;
    let mut guard = MemoryGuard {
        backend: args.device,
        dir: &args.run_dir,
        range: MemoryRange::default(),
        durable_step: None,
    };
    guard.check("startup", true)?;
    tch::manual_seed(0);
    let corpus = Corpus::load(&args.data_root)?;
    guard.check("before model allocation", true)?;
    let model = Dense::baseline(device);
    let mut opt = Adam::new(&model);
    let protocol = protocol(&args, &corpus, priority, identity);
    let mut state: State = if args.resume {
        let state: State = serde_json::from_slice(&opt.load(&latest)?)?;
        if state.protocol != protocol
            || state.step != opt.step
            || state.consumed_targets != opt.step * TARGETS
            || opt.step > args.stop_after
        {
            return Err("resume protocol/counter mismatch".into());
        }
        guard.durable_step = Some(state.step);
        guard.range.include(&state.memory);
        state
    } else {
        eprintln!("dense baseline: initial full-source validation");
        let initial = validate(&model, &corpus, args.microbatch, &mut guard)?;
        State {
            protocol,
            step: 0,
            consumed_targets: 0,
            initial_validation: initial,
            train_seconds: 0.,
            active_wall_seconds: 0.,
            train_curve: Vec::new(),
            memory: guard.range.clone(),
        }
    };
    let previous_wall = state.active_wall_seconds;
    if !args.resume {
        state.active_wall_seconds = start.elapsed().as_secs_f64();
        checkpoint(&args.run_dir, &opt, &state)?;
        guard.durable_step = Some(opt.step);
    }
    while opt.step < args.stop_after {
        let tick = Instant::now();
        let (ce, grad_norm) = train_update(&model, &mut opt, &corpus, args.microbatch, &mut guard)?;
        state.train_seconds += tick.elapsed().as_secs_f64();
        state.step = opt.step;
        state.consumed_targets = opt.step * TARGETS;
        if opt.step == 1 || opt.step.is_multiple_of(10) || opt.step == args.stop_after {
            eprintln!(
                "step {}/{} targets {} train CE {:.5}",
                opt.step, UPDATES, state.consumed_targets, ce
            );
            state.train_curve.push(
                json!({"step":opt.step,"train_ce":ce,"gradient_norm":grad_norm,
                                          "lr":dense_model::learning_rate(opt.step-1)}),
            );
        }
        if opt.step.is_multiple_of(args.save_every) || opt.step == args.stop_after {
            state.active_wall_seconds = previous_wall + start.elapsed().as_secs_f64();
            state.memory = guard.range.clone();
            checkpoint(&args.run_dir, &opt, &state)?;
            guard.durable_step = Some(opt.step);
        }
    }
    if opt.step < UPDATES {
        eprintln!(
            "partial run: checkpoint at step {}, {} targets",
            opt.step, state.consumed_targets
        );
        return Ok(());
    }
    eprintln!("dense baseline: final full-source validation");
    let final_eval = validate(&model, &corpus, args.microbatch, &mut guard)?;
    let tokens = state.consumed_targets as f64;
    // Dense projection/FFN/LM head multiplies plus QK and AV square attention.
    // Forward+backward uses three times the forward GEMM estimate.
    let flops_per_token = 6 * (12 * WIDTH * WIDTH + WIDTH * VOCAB) + 12 * (CONTEXT - 1) * WIDTH;
    let receipt = json!({"status":"complete","step":opt.step,"training_targets":state.consumed_targets,
        "trainable_parameters":model.count(), "protocol":state.protocol,
        "build":{"commit":env!("DRIVER_BUILD_COMMIT"),"dirty":env!("DRIVER_BUILD_DIRTY")=="true",
                 "rustc":env!("DRIVER_RUSTC"),"libtorch_intraop_threads":tch::get_num_threads()},
        "rocm_memory_bytes":(args.device==Backend::Rocm).then_some(&guard.range),
        "initial_validation":state.initial_validation,"final_validation":final_eval,
        "train_curve":state.train_curve,"training_seconds":state.train_seconds,
        "training_targets_per_second":tokens/state.train_seconds,
        "active_wall_seconds":previous_wall+start.elapsed().as_secs_f64(),
        "compute_proxy":{"train_gemm_flops_per_target":flops_per_token,
                        "training_gemm_flops":flops_per_token as u64 * state.consumed_targets as u64},
        "reference":{"run":"v3.1-L","parameters":98340864,"step":500,"targets":UPDATES*TARGETS,
                     "macro_main_ce":5.4000937836071925}});
    let mut staged = tempfile::NamedTempFile::new_in(&args.run_dir)?;
    serde_json::to_writer_pretty(&mut staged, &receipt)?;
    writeln!(staged)?;
    staged.as_file().sync_all()?;
    staged.persist_noclobber(report)?;
    File::open(&args.run_dir)?.sync_all()?;
    println!(
        "completed 500 updates and {} training targets",
        state.consumed_targets
    );
    Ok(())
}
