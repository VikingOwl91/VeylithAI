"""v3.1-PERF-3A read-only audit: map profiled elementwise/copy kernels to source.

Parses the existing PERF-1 trace (no GPU, no model, no training), aggregates
operator-level kernel time by shape and dtype, annotates the source path, and
writes the audit summary JSON with the candidate assessment.
"""

from __future__ import annotations

import os
import json
from collections import defaultdict

TRACE = os.path.expanduser("~/veylith-perf1-20261027/profile/trace2/trace.json")
OUT = "veylith/v3/docs/v31-perf3a-summary.json"
UPDATES = 5

# op signature -> (source, phase note)
SOURCE = {
    ("aten::mul", ((4088, 960), (4088, 960))): "model.py:109 SwiGLU silu(w1(x)) * w2(x)",
    ("aten::silu", ((4088, 960),)): "model.py:109 SwiGLU F.silu(w1(x))",
    ("aten::silu_backward", ((4088, 960), (4088, 960))): "model.py:109 autograd of SwiGLU silu",
    ("aten::native_dropout", ((4088, 512), (), ())): "model.py:110 SwiGLU expert dropout",
    ("aten::native_dropout_backward", ((4088, 512), (4088, 512), ())): "model.py:110 teacher-forced dropout backward",
    ("aten::mul", ((4088, 1), (4088, 512))): "model.py:148 weights[:, e] * expert(flat)",
    ("aten::mul", ((4088, 512), (4088, 1))): "model.py:148 backward: grad * weights",
    ("aten::mul", ((4088, 512), (4088, 512))): "model.py:148 backward: grad * expert output (weight grad)",
    ("aten::add", ((4088, 512), (4088, 512), ())): "model.py:149 out = out + contribution (FP32 combine)",
    ("aten::add_", ((4088, 512), (4088, 512), ())): "autograd accumulation of combine output gradient",
    ("aten::sum", ((4088, 512), (), (), ())): "autograd reduction for per-token combine weights",
    ("aten::copy_", ((4088, 512), (4088, 512), ())): "autocast FP32->BF16 cast / residual promotion",
    ("aten::add_", ((960, 512), (960, 512), ())): "AccumulateGrad expert w1/w2 gradient (FP32)",
    ("aten::add_", ((512, 960), (512, 960), ())): "AccumulateGrad expert w3 gradient (FP32)",
    ("aten::add_", ((512, 512), (512, 512), ())): "AccumulateGrad attention projection gradient",
    ("aten::add_", ((8192, 512), (8192, 512), ())): "AccumulateGrad LM head gradient",
    ("aten::pow", ((8, 511, 512), ())): "model.py:25 RMSNorm x.pow(2) (FP32 residual)",
    ("aten::mean", ((8, 511, 512), (), (), ())): "model.py:25 RMSNorm mean",
    ("aten::rsqrt", ((8, 511, 1),)): "model.py:25 RMSNorm rsqrt",
    ("aten::mul", ((8, 511, 512), (512,))): "model.py:25 RMSNorm * weight (FP32)",
    ("aten::mul", ((8, 511, 512), (8, 511, 1))): "model.py:25 RMSNorm backward",
    ("aten::embedding_dense_backward", ((8, 511, 512), (8, 511), (), (), ())): "model.py:192 tok_emb backward",
    ("aten::cat", ((8, 8, 511, 32), (8, 8, 511, 32))): "model.py:36 apply_rope cat",
    ("aten::copy_", ((4088, 8192), (4088, 8192), ())): "CE loss logits FP32/BF16 boundary",
    ("aten::fill_", ((4088, 512), ())): "unattributed activation buffer zero-init",
}


def freeze(value):
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def main():
    with open(TRACE, encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]

    kernel_time = defaultdict(float)
    kernel_names = defaultdict(float)
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy"):
            continue
        external = (event.get("args") or {}).get("External id")
        if external is not None:
            kernel_time[external] += float(event.get("dur", 0))
            kernel_names[event.get("name", "")] += float(event.get("dur", 0))

    annotations = [event for event in events
                   if event.get("cat") == "user_annotation"
                   and event.get("name") in ("data_fetch", "forward", "backward",
                                             "grad_clip", "optimizer", "update")]
    intervals = [(event["name"], float(event["ts"]),
                  float(event["ts"]) + float(event["dur"])) for event in annotations]
    priority = {"data_fetch": 0, "forward": 1, "backward": 1,
                "optimizer": 2, "grad_clip": 2, "update": 3}

    def phase(timestamp):
        best = None
        for name, low, high in intervals:
            if low <= timestamp <= high:
                if (best is None or priority[name] < priority[best[0]]
                        or (priority[name] == priority[best[0]]
                            and high - low < best[2] - best[1])):
                    best = (name, low, high)
        return best[0] if best else "none"

    aggregate = defaultdict(lambda: [0, 0.0, defaultdict(int), set()])
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        args = event.get("args") or {}
        dims = freeze(args.get("Input Dims") or [])
        if not dims:
            continue
        types = freeze(args.get("Input type") or [])
        key = (event.get("name"), dims, types)
        external = args.get("External id")
        aggregate[key][0] += 1
        aggregate[key][1] += kernel_time.get(external, 0.0)
        aggregate[key][2][phase(float(event.get("ts", 0)))] += 1

    rows = []
    for (name, dims, types), (count, microseconds, phases, _) in sorted(
            aggregate.items(), key=lambda item: -item[1][1]):
        if microseconds / UPDATES / 1000 < 0.3:
            continue
        source = SOURCE.get((name, dims), "")
        rows.append({
            "op": name,
            "dims": dims,
            "types": types,
            "calls_per_update": count / UPDATES,
            "ms_per_update": microseconds / UPDATES / 1000,
            "phases": {key: value / UPDATES for key, value in phases.items()},
            "source": source,
        })
        if len(rows) >= 28:
            break

    top_kernels = [
        {"kernel": name[:140], "ms_per_update": value / UPDATES / 1000}
        for name, value in sorted(kernel_names.items(), key=lambda item: -item[1])[:20]
    ]

    summary = {
        "meta": {
            "experiment": "v3.1-PERF-3A elementwise and dtype fusion audit",
            "mode": "read-only; no GPU workloads, no benchmarks, no code changes",
            "git_head": "5a72192",
            "trace": TRACE,
            "trace_updates": UPDATES,
        },
        "perf1_context": {
            "elementwise_ms": 309.1,
            "copies_ms": 81.8,
            "gemm_ms": 245.0,
            "kernels_per_update": 41016,
        },
        "promotion_finding": {
            "summary": "The residual stream is FP32: the embedding output is FP32, "
                       "RMSNorm multiplies by an FP32 weight and returns FP32, and the "
                       "MoE dense combine accumulates in FP32. Autocast therefore casts "
                       "FP32 activations to BF16 before every Linear.",
            "evidence": [
                "aten::pow/mean/rsqrt on (8,511,512) and (8,511,1) with type float",
                "aten::embedding_dense_backward output type float",
                "aten::add (8,511,512) with float, bfloat16 operands in forward",
                "aten::copy_ (4088,512) bf16<-float: 1400 calls/update, 21.83 ms",
                "aten::copy_ (4088,512) float<-bf16: 952 calls/update, 11.08 ms",
            ],
            "norm_consumers_per_block": 20,
            "blocks": 8,
            "microbatches": 8,
            "norm_consumer_casts_per_update": 1280,
            "single_cast_replacement_per_update": 128,
        },
        "evidence": rows,
        "top_kernels": top_kernels,
        "perf1_items": [
            {"item": "FP32 gradient accumulation", "perf1_ms": 71.3,
             "actual": "Mixture of parameter-gradient adds (about 15-16 ms), activation-"
                       "gradient adds (about 35 ms) and combine/residual FP32 adds "
                       "(about 24 ms).",
             "source": "autograd AccumulateGrad plus model.py:149/181",
             "avoidable": "launch overhead only for parameter adds; activation adds are "
                          "required reductions"},
            {"item": "BF16 expert-weight multiplication", "perf1_ms": 36.7,
             "actual": "SwiGLU silu(w1) * w2 at (4088,960), 448 forward and 896 backward "
                       "calls; not the router weighting.",
             "source": "model.py:109", "avoidable": "fusable activation; ~12-18 ms"},
            {"item": "FP32 binary operations", "perf1_ms": 32.8,
             "actual": "Combine adds 16.27 ms, RMSNorm muls 7.9 ms, attention RoPE and "
                       "softmax muls, residual adds.",
             "source": "model.py:149, 25, 36", "avoidable": "depends on the sub-op"},
            {"item": "BF16 copies", "perf1_ms": 35.4,
             "actual": "FP32->BF16 autocast casts of norm outputs to every Linear "
                       "(21.83 ms, 1400 calls), CE boundary 3.78 ms, attention reshapes.",
             "source": "precision.py autocast plus model.py RMSNorm", "avoidable": "~18 ms"},
            {"item": "BF16-to-FP32 conversion", "perf1_ms": 27.4,
             "actual": "Residual/combine promotions 11.08 ms, weight-gradient casts to "
                       "FP32 8.4 ms, CE cast 3.17 ms.",
             "source": "model.py:181, autograd", "avoidable": "only if the residual is "
                                                              "kept in BF16 (numerical change)"},
            {"item": "Manual-unroll operations", "perf1_ms": 27.5,
             "actual": "Broadcast multiplies: router weights (12.28 + 6.77 ms), RMSNorm "
                       "weight mul (7.9 ms), RoPE/attention muls.",
             "source": "model.py:148, 25, 36", "avoidable": "fusable with the combine or "
                                                             "norm changes"},
        ],
        "candidates": [
            {"id": "P1", "name": "Single-cast at RMSNorm output",
             "targets": ["BF16 copies", "FP32 binary", "Manual-unroll norm muls"],
             "measured_cluster_ms": 21.83,
             "attributable_ms": 19.96,
             "max_saving_ms": 18.0,
             "assumptions": "1,280 of the 1,400 (4088,512) FP32->BF16 casts are norm "
                            "consumers; cast time scales with elements; the combine "
                            "accumulator is kept FP32 and the residual stays FP32; "
                            "autocast cast equals .to(bfloat16).",
             "complexity": "low: RMSNorm returns a BF16 copy of the same FP32 result; "
                           "MoE zeros/accumulator made explicitly FP32.",
             "numerical_risk": "none expected: the GEMM operands are the same BF16 values "
                               "as the per-consumer cast; combine and residual unchanged.",
             "autograd_risk": "none: no change to graph structure or parameter dtypes.",
             "rocm": "none; no new kernels.",
             "correctness_test": "unit: RMSNorm FP32 result cast to BF16 equals the old "
                                  "consumer cast; GPU 2-update loss/gradient comparison "
                                  "with bitwise equality as the target.",
             "perf_test": "5-update profiler; count copy_ (4088,512) FP32->BF16 and "
                          "elementwise ms.",
             "expected_update_pct": -2.3},
            {"id": "P2", "name": "Fused MoE weighted aggregation",
             "targets": ["FP32 binary", "Manual-unroll", "BF16 copies"],
             "measured_cluster_ms": 79.31,
             "attributable_ms": 79.31,
             "max_saving_ms": 15.0,
             "assumptions": "Fusion removes the FP32 contribution and d_w intermediates "
                            "(about 7.5 GB/update each) and kernel launches, keeping FP32 "
                            "accumulation and separate mul/add rounding.",
             "complexity": "medium-high: custom autograd function or Triton kernel for "
                           "forward combine and both gradient paths.",
             "numerical_risk": "reduction order for d_w and FMA vs separate mul/add; "
                               "requires an explicit no-FMA or documented-ulp choice.",
             "autograd_risk": "custom backward must match three gradient paths exactly.",
             "rocm": "Triton 3.8 present; deterministic reductions, no atomics.",
             "correctness_test": "synthetic combine at (448 calls, 4088x512) against the "
                                  "reference outputs and d_expert/d_weights; tolerance 0 or "
                                  "documented ulp.",
             "perf_test": "isolated combine microbenchmark; kernel count and ms/update.",
             "expected_update_pct": -1.9},
            {"id": "P3", "name": "SwiGLU activation fusion (silu + mul)",
             "targets": ["BF16 expert-weight multiplication", "Manual-unroll"],
             "measured_cluster_ms": 58.91,
             "attributable_ms": 58.91,
             "max_saving_ms": 14.0,
             "assumptions": "Removing the silu intermediate (7.85 MB BF16 per call, 448 "
                            "forward and 896 backward paths) saves write+read traffic; "
                            "same op order and rounding.",
             "complexity": "medium: Triton kernel or scoped torch.compile.",
             "numerical_risk": "low if the fused kernel preserves silu then multiply and "
                               "the exact backward expressions.",
             "autograd_risk": "custom backward for silu and w2 paths.",
             "rocm": "Triton 3.8; Inductor path optional.",
             "correctness_test": "SwiGLU forward/backward vs reference at (4088,960); "
                                  "tolerance 0.",
             "perf_test": "isolated SwiGLU microbenchmark; profile ms/update.",
             "expected_update_pct": -1.5},
            {"id": "P4", "name": "Multi-tensor gradient accumulation",
             "targets": ["FP32 gradient accumulation"],
             "measured_cluster_ms": 24.0,
             "attributable_ms": 15.5,
             "max_saving_ms": 7.0,
             "assumptions": "Parameter-sized adds are bandwidth-bound; only launch "
                            "overhead (about 1,832 adds/update) is removed.",
             "complexity": "medium: autograd AccumulateGrad is internal; needs a custom "
                           "buffer or post-backward foreach accumulation.",
             "numerical_risk": "same per-tensor add order, low.",
             "autograd_risk": "moderate: changing accumulation timing/ownership.",
             "rocm": "foreach kernels already used by AdamW.",
             "correctness_test": "gradient equality against the reference for two "
                                  "microbatches.",
             "perf_test": "count add_ launches and parameter-grad ms/update.",
             "expected_update_pct": -0.9},
            {"id": "P5", "name": "Scoped torch.compile / Inductor",
             "targets": ["all elementwise clusters"],
             "measured_cluster_ms": None,
             "attributable_ms": None,
             "max_saving_ms": None,
             "assumptions": "Fusion succeeds without graph breaks; determinism preserved.",
             "complexity": "high: topk, one_hot, bincount and .cpu() cause graph breaks; "
                           "compile time and recompiles.",
             "numerical_risk": "Inductor may reorder reductions; needs validation.",
             "autograd_risk": "moderate.",
             "rocm": "Triton-based Inductor on gfx1201; unproven for this model.",
             "correctness_test": "bitwise/ulp comparison on 2 updates.",
             "perf_test": "5-update profile after warm compilation.",
             "expected_update_pct": None},
            {"id": "P6", "name": "Custom Triton kernels (later option)",
             "targets": ["P2", "P3"],
             "measured_cluster_ms": None,
             "attributable_ms": None,
             "max_saving_ms": None,
             "assumptions": "Only as the implementation vehicle for P2/P3 after a "
                            "numerically neutral prototype.",
             "complexity": "high", "numerical_risk": "kernel-specific",
             "autograd_risk": "high", "rocm": "Triton 3.8",
             "correctness_test": "per-kernel reference comparison",
             "perf_test": "isolated plus end-to-end profile",
             "expected_update_pct": None},
        ],
        "ranking": [
            {"rank": 1, "id": "P1", "why": "largest purely avoidable cluster, "
                                           "numerically neutral, lowest complexity"},
            {"rank": 2, "id": "P2", "why": "largest measured cluster but only 15 ms "
                                           "avoidable and precision-sensitive"},
            {"rank": 3, "id": "P3", "why": "clear middle cluster, exact fusion possible"},
            {"rank": 4, "id": "P4", "why": "small launch-overhead-only gain"},
            {"rank": 5, "id": "P5", "why": "uncertain payoff, high complexity"},
            {"rank": 6, "id": "P6", "why": "implementation vehicle, not a first step"},
        ],
        "selected": {
            "id": "P1",
            "name": "Single-cast at RMSNorm output",
            "rationale": "18 ms of measured cast traffic is pure overhead; the GEMM "
                         "operands remain the same BF16 values, the combine and residual "
                         "stay FP32, and no new kernels or dependencies are required.",
            "expected_update_ms": -18.0,
            "expected_update_pct": -2.3,
            "bounded_prototype": "PERF-3B: change RMSNorm to return a BF16 cast of the "
                                 "same FP32 result, force the MoE accumulator to FP32, "
                                 "then run a 2-update numerical equivalence check and a "
                                 "5-update profile.",
        },
        "unknowns": [
            "About 120 of the 1,400 FP32->BF16 (4088,512) casts are not attributed to "
            "the 20 norm consumers per block; they may be attention or CE boundary "
            "casts.",
            "aten::fill_ (4088,512) at 8.63 ms BF16 and 6.68 ms FP32 per update has no "
            "stack trace; its source is not provable from this trace.",
            "Op-to-source mapping uses shape, dtype, phase and model semantics; the "
            "trace has no stacks (with_stack=False).",
            "The 309/82/245 category figures come from PERF-1 kernel attribution; "
            "op-level sums can double count nested ops.",
            "Savings assume the elementwise stream is the critical path and that "
            "removing casts does not shift time into the GEMMs.",
        ],
        "integrity": {
            "gpu_work": "none",
            "code_changes": "none",
            "drift_guard": "OK",
            "kfd": "idle",
        },
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
