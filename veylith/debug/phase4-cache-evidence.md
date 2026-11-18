# VEYLITH-3 · Phase 4 E16/K4 build and workspace-cache evidence

The scaling report identifies the Phase 4 environment as PyTorch `2.16.0.dev20261006+rocm7.14` on the RX 9070 XT. The effective value of `TORCH_CUBLAS_WORKSPACE_CACHE` for the failing sparse process is **unknown**. The saved evidence supports keeping sparse E16/K4 out of training pending diagnosis. [Scaling report](phase4-scaling.md)

| Question | What the saved run establishes |
|---|---|
| Which interpreter and build? | The [launcher](https://github.com/VikingOwl91/VeylithAI/blob/be68c2c32e9629a14a5a6ae8329ca214ea639044/veylith/debug/version-isolation/run-scaling.sh) selects the Phase 4 interpreter. A warning in the [sparse log](logs/scaling_b64_e16k4_sparse.log) identifies the same environment's PyTorch installation. The version comes from the report. The failing process printed neither a wheel version nor a Git revision. |
| Which cache setting? | The launcher passes the caller's environment through. Neither the launcher nor the [benchmark](https://github.com/VikingOwl91/VeylithAI/blob/be68c2c32e9629a14a5a6ae8329ca214ea639044/veylith/debug/benchmark.py) sets or records the variable. The log cannot distinguish an inherited override from an unset variable. |
| What happened? | The [grid](version-isolation/scaling-results.txt) records one sparse exit 134 at batch 64 × 256, 16 experts, top-4. The sparse log reports a GPU page fault during a GEMM. The paired [dense run](logs/scaling_b64_e16k4_dense.log) completed at 215.61 ms per step. |

## Comparison with the cache experiments

EXP1–3 used eight experts, top-2 routing and the expert-gradient filter at batch 64 × 256. The experiment report assigns EXP1 to testE (August 24 nightly) with caching disabled, EXP2 to testD (October 5 nightly) with caching enabled, and EXP3 to testE with a DMA-copy setting. All three trials in each group logged GPU memory faults. EXP3 supplies no independent cache-setting measurement. [Experiment settings and builds](phase4-root-cause.md) · [EXP1 log](version-isolation/logs/exp1_cache0_1.log) · [EXP2 log](version-isolation/logs/exp2_cache1_1.log) · [EXP3 log](version-isolation/logs/exp3_dmacopy_1.log)

The earlier results make the cache path a hypothesis for the scaling fault. The experiments used different builds and workloads, and the scaling run lacks a process-level environment record. A common fault class alone cannot identify the underlying mechanism. Resolving the historical setting would require a contemporaneous launch record tied to that run. Inspecting a present-day shell or reinstalling the wheel cannot recover it.

The sibling branch `v3/data-pipeline` at `113df4d` contains a separate instrumented investigation in `veylith/v3/docs/v31-perf5q-summary.json`. Its 5P, 5Q-A and 5Q-B runs all completed without a fault while recording the reported dispatch geometry. The summary identifies its experiment as v3.1-PERF-5Q and its run revision as `d73b105`. A search of the sibling commit's tracked files found no contemporaneous Phase 4 launch record identifying the E16/K4 scaling process's cache setting.
