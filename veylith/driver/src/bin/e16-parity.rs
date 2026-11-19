//! One CPU training step on the archived, synthetic 16-expert test vector.
use clap::Parser;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tch::nn::{self, OptimizerConfig};
use tch::{Device, Kind, Tensor};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Tensors = BTreeMap<String, Tensor>;
const FIXTURE_SHA: &str = "1e44a0b122f14e28efd8011e577cc912f128f22e252e7517b04200cc50d930da";
const LOSS_ATOL: f64 = 1e-6;
const GRAD_ATOL: f64 = 1e-6;
const CHANGE_ATOL: f64 = 2e-6;
const EXPERTS: i64 = 16;
const K: i64 = 4;

#[derive(Parser)]
#[command(
    version,
    about = "Compare a tiny E16/K4 CPU training step with the frozen Python result"
)]
struct Args {
    /// The published e16-k4-cpu.npz synthetic test vector.
    #[arg(long)]
    fixture: PathBuf,
    /// Write a new JSON report after both implementations pass.
    #[arg(long)]
    out: PathBuf,
}

struct SwiGlu<'a> {
    w1: &'a Tensor,
    w2: &'a Tensor,
    w3: &'a Tensor,
}

impl<'a> SwiGlu<'a> {
    fn new(params: &'a Tensors, prefix: &str) -> Self {
        Self {
            w1: &params[&format!("{prefix}.w1.weight")],
            w2: &params[&format!("{prefix}.w2.weight")],
            w3: &params[&format!("{prefix}.w3.weight")],
        }
    }

    fn apply(&self, x: &Tensor) -> Tensor {
        (x.matmul(&self.w1.tr()).silu() * x.matmul(&self.w2.tr())).matmul(&self.w3.tr())
    }
}

#[derive(Clone, Copy)]
enum Execution {
    AllExperts,
    SelectedPerToken,
}

fn forward(params: &Tensors, tokens: &Tensor, mode: Execution) -> Result<(Tensor, Tensor)> {
    let steps = tokens.size()[1] - 1;
    let x = Tensor::embedding(
        &params["embed.weight"],
        &tokens.narrow(1, 0, steps),
        -1,
        false,
        false,
    )
    .reshape([-1, 8]);
    let targets = tokens.narrow(1, 1, steps).reshape([-1]);
    let logits = x.matmul(&params["moe.gate.weight"].tr());
    let (scores, ids) = logits.topk(K, -1, true, true);
    let probs = scores.softmax(-1, Kind::Float);
    let assignments = ids.one_hot(EXPERTS).to_kind(Kind::Float);
    let fractions = assignments.mean_dim(&[0i64, 1][..], false, Kind::Float);
    let gate_mean = logits
        .softmax(-1, Kind::Float)
        .mean_dim(&[0i64][..], false, Kind::Float);
    let auxiliary = (fractions * gate_mean).sum(Kind::Float) * EXPERTS;
    let shared = SwiGlu::new(params, "moe.shared").apply(&x);
    let experts: Vec<_> = (0..EXPERTS)
        .map(|e| SwiGlu::new(params, &format!("moe.experts.{e}")))
        .collect();
    let routed = match mode {
        Execution::AllExperts => {
            let weights = (assignments * probs.unsqueeze(-1)).sum_dim_intlist(
                &[1i64][..],
                false,
                Kind::Float,
            );
            let mut y = shared;
            for (e, expert) in experts.iter().enumerate() {
                // Use an out-of-place sum, as in the archived differentiable path.
                y = &y + weights.narrow(1, e as i64, 1) * expert.apply(&x);
            }
            y
        }
        Execution::SelectedPerToken => {
            // A scalar CPU reference. Each input visits just its four selected experts.
            let selections = Vec::<i64>::try_from(ids.reshape([-1]))?;
            let mut outputs = Vec::new();
            for (row, selected) in selections.chunks_exact(K as usize).enumerate() {
                let x_row = x.narrow(0, row as i64, 1);
                let mut y = shared.narrow(0, row as i64, 1);
                for (rank, &e) in selected.iter().enumerate() {
                    let w = probs.get(row as i64).get(rank as i64);
                    y = &y + w * experts[e as usize].apply(&x_row);
                }
                outputs.push(y);
            }
            Tensor::cat(&outputs, 0)
        }
    };
    let output_logits = (x + routed).matmul(&params["head.weight"].tr());
    let ce = output_logits.cross_entropy_for_logits(&targets);
    let total = &ce + &auxiliary * 0.01;
    Ok((Tensor::stack(&[ce, auxiliary, total], 0), ids))
}

fn check_error(measured: &Tensor, expected: &Tensor, atol: f64, label: &str) -> Result<f64> {
    if measured.size() != expected.size() || measured.kind() != expected.kind() {
        return Err(format!("{label} shape or dtype mismatch").into());
    }
    let diff = (measured - expected).abs().max().double_value(&[]);
    if !diff.is_finite() || diff > atol {
        return Err(format!("{label} max absolute error {diff} exceeds {atol}").into());
    }
    Ok(diff)
}

fn compare(reference: &Tensors, mode: Execution) -> Result<Value> {
    let vars = nn::VarStore::new(Device::Cpu);
    let mut params = Tensors::new();
    for (key, value) in reference {
        if let Some(name) = key.strip_prefix("initial.") {
            let parameter = vars.root().var_copy(&format!("p{}", params.len()), value);
            params.insert(name.to_owned(), parameter);
        }
    }
    let mut optimizer = nn::AdamW::default().build(&vars, 0.001)?;
    let (losses, ids) = forward(&params, &reference["tokens"], mode)?;
    if !ids.equal(&reference["routes"]) {
        return Err("top-4 expert IDs or their order differ".into());
    }
    let loss_error = check_error(
        &losses,
        &reference["losses"],
        LOSS_ATOL,
        "CE, auxiliary and total losses",
    )?;
    losses.get(2).backward();
    let mut gradient_error = 0.0_f64;
    for (name, param) in &params {
        let key = format!("gradient.{name}");
        gradient_error = gradient_error.max(check_error(
            &param.grad(),
            &reference[&key],
            GRAD_ATOL,
            &key,
        )?);
    }
    optimizer.step();
    let mut update_error = 0.0_f64;
    for (name, param) in &params {
        let change = param.detach() - &reference[&format!("initial.{name}")];
        let key = format!("change.{name}");
        update_error = update_error.max(check_error(&change, &reference[&key], CHANGE_ATOL, &key)?);
    }
    let routes = Vec::<i64>::try_from(ids.one_hot(EXPERTS).sum_dim_intlist(
        &[0i64, 1][..],
        false,
        Kind::Int64,
    ))?;
    if routes.contains(&0) {
        return Err("fixture did not exercise every expert".into());
    }
    Ok(json!({
        "parameters": params.values().map(Tensor::numel).sum::<usize>(), "parameter_tensors": params.len(),
        "targets": ids.size()[0], "route_counts": routes, "ordered_top4_exact": true,
        "cross_entropy": losses.double_value(&[0]), "auxiliary": losses.double_value(&[1]),
        "objective": losses.double_value(&[2]), "max_abs_loss_error": loss_error,
        "max_abs_gradient_error": gradient_error, "max_abs_parameter_change_error": update_error,
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.out.try_exists()? {
        return Err("report already exists".into());
    }
    if format!("{:x}", Sha256::digest(fs::read(&args.fixture)?)) != FIXTURE_SHA {
        return Err("synthetic fixture SHA-256 mismatch".into());
    }
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    let reference: Tensors = Tensor::read_npz(&args.fixture)?.into_iter().collect();
    let dense = compare(&reference, Execution::AllExperts)?;
    let selected = compare(&reference, Execution::SelectedPerToken)?;
    let report = json!({
        "experiment":"VEYLITH-6 synthetic CPU one-step parity", "dtype":"float32", "threads":1,
        "fixture_sha256":FIXTURE_SHA, "archive_commit":"be68c2c32e9629a14a5a6ae8329ca214ea639044",
        "archive_source":"veylith/v2/train.py", "archive_source_git_blob":"bcd5eb60524a5ec15089fd9fdfca09e3a2761e8f",
        "source_sha256":format!("{:x}", Sha256::digest(include_bytes!("e16-parity.rs"))),
        "build_commit":env!("DRIVER_BUILD_COMMIT"), "build_dirty":env!("DRIVER_BUILD_DIRTY") == "true",
        "rustc":env!("DRIVER_RUSTC"), "tch":"0.27.0", "libtorch_target":"2.15.0",
        "tolerances":{"loss":LOSS_ATOL,"gradient":GRAD_ATOL,"parameter_change":CHANGE_ATOL},
        "dense_combined":dense,"selected_per_token":selected
    });
    let parent = args
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    file.flush()?;
    file.persist_noclobber(args.out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_cpu_paths_match_the_archived_step_and_detect_changed_gradient() -> Result<()> {
        tch::set_num_threads(1);
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/e16-k4-cpu.npz");
        let mut reference: Tensors = Tensor::read_npz(fixture)?.into_iter().collect();
        compare(&reference, Execution::AllExperts)?;
        compare(&reference, Execution::SelectedPerToken)?;
        let key = "gradient.moe.gate.weight";
        reference.insert(key.into(), &reference[key] + 0.01);
        let failure = compare(&reference, Execution::AllExperts).unwrap_err();
        assert!(failure.to_string().contains(key));
        Ok(())
    }
}
