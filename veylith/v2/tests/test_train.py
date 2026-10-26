"""CPU-only tests for checkpointing, resume, loss components and evaluation.

Run from the repository root:

    python -m pytest veylith/v2/tests -q

Everything here is forced onto the CPU (``--device cpu`` or ``pick_device``
called directly); no test touches a GPU.
"""

import contextlib
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
if V2 not in sys.path:
    sys.path.insert(0, V2)

import train as T  # noqa: E402

VOCAB = 64


@pytest.fixture(autouse=True, scope="session")
def cpu_only():
    """Fail the session if any test initialized a GPU context."""
    yield
    assert not torch.cuda.is_initialized(), "a CPU test initialized CUDA"


def make_cfg(**over):
    kw = dict(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, block_size=32,
              sliding_window=16, full_attn_every=4, dropout=0.1,
              n_experts=4, n_experts_per_tok=2, first_k_dense=1,
              n_mtp=1, mtp_weight=0.5, moe_aux_weight=0.01, vocab_size=VOCAB)
    kw.update(over)
    return T.Config(**kw)


def make_args(checkpoint, **over):
    args = T.parse_args([
        "--device", "cpu", "--tokenizer", "char",
        "--n_layer", "2", "--n_head", "2", "--n_kv_head", "1", "--n_embd", "32",
        "--block_size", "32", "--sliding_window", "16", "--n_experts", "4",
        "--n_experts_per_tok", "2", "--first_k_dense", "1",
        "--max_iters", "12", "--batch_size", "4", "--eval_interval", "3",
        "--eval_iters", "2", "--warmup_iters", "2", "--checkpoint", str(checkpoint),
    ])
    for key, value in over.items():
        setattr(args, key, value)
    return args


def make_data(seed=0, n=4000):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=generator)


def fresh_model(cfg, args, init_seed=7):
    torch.manual_seed(init_seed)
    model = T.GPT(cfg)
    return model, torch.optim.AdamW(model.parameters(), lr=args.lr)


def train_once(model, optimizer, args, data, start_step=0):
    return T.run_training(
        model, optimizer, data, data, args, "cpu",
        contextlib.nullcontext(), {"type": "char"}, start_step=start_step)


def assert_same_model(left, right):
    a, b = left.state_dict(), right.state_dict()
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), f"model tensor differs: {key}"


def assert_same_optimizer(left, right):
    a, b = left.state_dict(), right.state_dict()
    assert a["param_groups"] == b["param_groups"]
    assert set(a["state"]) == set(b["state"])
    for index in a["state"]:
        for key, va in a["state"][index].items():
            vb = b["state"][index][key]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), f"optimizer state differs: {index}.{key}"
            else:
                assert va == vb, f"optimizer state differs: {index}.{key}"


def test_resume_reproduces_uninterrupted_run(tmp_path):
    """12 iterations in one go == 6 iterations, state, resume, 6 more."""
    cfg = make_cfg()
    data = make_data()

    ref_args = make_args(tmp_path / "ref" / "ckpt.pt")
    ref_model, ref_opt = fresh_model(cfg, ref_args)
    torch.manual_seed(99)
    _, _, ref_losses = train_once(ref_model, ref_opt, ref_args, data)
    assert len(ref_losses) == 12

    part_args = make_args(tmp_path / "part" / "ckpt.pt", stop_after=6)
    part_model, part_opt = fresh_model(cfg, part_args)
    torch.manual_seed(99)
    _, _, first_losses = train_once(part_model, part_opt, part_args, data)
    assert len(first_losses) == 6

    state_path = T.default_state_path(part_args.checkpoint)
    state = T.load_train_state(state_path, cfg, part_args, {"type": "char"})
    assert state["kind"] == "train_state"
    assert state["format"] == T.STATE_FORMAT
    assert state["step"] == 6
    for key in ("model", "optimizer", "best_val", "best_model", "config", "rng",
                "fingerprint", "args"):
        assert key in state
    assert "torch" in state["rng"]

    resume_args = make_args(tmp_path / "part" / "ckpt.pt")
    resume_model, resume_opt = fresh_model(cfg, resume_args, init_seed=12345)
    resume_model.load_state_dict(state["model"])
    resume_opt.load_state_dict(state["optimizer"])
    T.restore_rng_state(state["rng"])
    _, _, second_losses = train_once(resume_model, resume_opt, resume_args, data,
                                     start_step=state["step"])

    assert second_losses, "resume ran no iterations"
    assert ref_losses == first_losses + second_losses, "loss sequence diverged on resume"
    assert_same_model(ref_model, resume_model)
    assert_same_optimizer(ref_opt, resume_opt)


def test_atomic_save_accepts_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    T.atomic_save({"value": 1}, "checkpoint.pt")
    assert (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "checkpoint.pt.tmp").exists()
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=False)["value"] == 1


def test_failed_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pt"
    T.atomic_save({"value": 1}, str(path))

    def broken_save(obj, handle):
        handle.write(b"partial bytes")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(RuntimeError):
        T.atomic_save({"value": 2}, str(path))
    monkeypatch.undo()

    assert torch.load(path, weights_only=False)["value"] == 1
    assert not (tmp_path / "ckpt.pt.tmp").exists()


def test_config_mismatch_is_rejected(tmp_path):
    cfg = make_cfg()
    args = make_args(tmp_path / "ckpt.pt", stop_after=3)
    data = make_data()
    model, opt = fresh_model(cfg, args)
    torch.manual_seed(99)
    train_once(model, opt, args, data)
    path = T.default_state_path(args.checkpoint)

    with pytest.raises(ValueError) as err:
        T.load_train_state(path, make_cfg(n_embd=64), make_args(tmp_path / "ckpt.pt"),
                           {"type": "char"})
    assert "n_embd" in str(err.value)

    other = make_args(tmp_path / "ckpt.pt")
    other.batch_size = 8
    with pytest.raises(ValueError) as err:
        T.load_train_state(path, cfg, other, {"type": "char"})
    assert "batch_size" in str(err.value)


def test_mode_artifact_is_not_a_resume_state(tmp_path):
    cfg = make_cfg()
    artifact = tmp_path / "model.pt"
    T.atomic_save({"kind": "model", "model": T.GPT(cfg).state_dict()}, artifact)
    with pytest.raises(ValueError):
        T.load_train_state(artifact, cfg, make_args(tmp_path / "x.pt"), {"type": "char"})


def test_legacy_inference_checkpoint_still_loads():
    path = os.path.join(V2, "checkpoints", "ckpt.pt")
    if not os.path.exists(path):
        pytest.skip("no trained checkpoint in this worktree")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert {"model", "config", "tokenizer", "val_loss"} <= set(ckpt)
    model = T.GPT(T.Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    assert torch.isfinite(torch.tensor(ckpt["val_loss"]))


def test_loss_composition_uses_mtp_shift():
    """loss == main CE(t+1) + w_mtp * CE(t+2) + w_aux * aux, with the t+2 shift."""
    cfg = make_cfg()
    torch.manual_seed(3)
    model = T.GPT(cfg).eval()
    x = torch.randint(0, VOCAB, (2, 32))
    y = torch.randint(0, VOCAB, (2, 32))
    logits, loss = model(x, y)

    main = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    mtp = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), y[:, 1:].reshape(-1))
    hidden = model.tok_emb(x)
    aux = x.new_zeros(())
    for block in model.blocks:
        hidden, block_aux = block(hidden)
        aux = aux + block_aux
    expected = main + cfg.mtp_weight * mtp + cfg.moe_aux_weight * aux

    assert torch.allclose(loss, expected, atol=1e-6), (loss.item(), expected.item())


@pytest.mark.parametrize("over", [
    {"n_embd": 33},
    {"n_head": 8, "n_kv_head": 3},
    {"n_embd": 34, "n_head": 4},
])
def test_config_invariants_are_enforced(over):
    with pytest.raises(AssertionError):
        T.GPT(make_cfg(**over))


def test_device_flag_forces_cpu():
    assert T.pick_device("cpu") == "cpu"
    args = T.parse_args(["--device", "cpu"])
    assert args.device == "cpu"
    assert args.resume is None and args.save_every is None


def test_cli_pause_resume_and_bare_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tinyshakespeare.txt").write_text(
        "ROMEO: first citizen: before we proceed any further, hear me speak.\n" * 400,
        encoding="utf-8")

    base = [
        sys.executable, os.path.join(V2, "train.py"), "--device", "cpu",
        "--tokenizer", "char", "--n_layer", "1", "--n_head", "2", "--n_kv_head", "1",
        "--n_embd", "32", "--block_size", "32", "--sliding_window", "16",
        "--n_experts", "4", "--n_experts_per_tok", "2", "--first_k_dense", "1",
        "--batch_size", "4", "--max_iters", "4", "--eval_interval", "2",
        "--eval_iters", "1", "--warmup_iters", "1", "--dropout", "0.0",
        "--save-every", "0", "--checkpoint", "checkpoint.pt",
    ]

    missing = subprocess.run(base + ["--resume"], cwd=tmp_path,
                             capture_output=True, text=True, timeout=180)
    assert missing.returncode != 0
    assert "no training state" in (missing.stderr + missing.stdout)

    paused = subprocess.run(base + ["--stop-after", "2"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=180)
    assert paused.returncode == 0, paused.stderr
    assert not (tmp_path / "checkpoint.pt").exists(), "a paused run must not write the artifact"
    assert (tmp_path / "checkpoint.state.pt").exists()

    resumed = subprocess.run(base + ["--resume"], cwd=tmp_path,
                             capture_output=True, text=True, timeout=180)
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed" in resumed.stdout
    assert (tmp_path / "checkpoint.pt").exists()
    assert "Traceback" not in resumed.stderr


# ------------------------------------------------ loss components / evaluation

def test_forward_metrics_recompose_the_composite():
    cfg = make_cfg()
    torch.manual_seed(5)
    model = T.GPT(cfg).eval()
    x = torch.randint(0, VOCAB, (3, 32))
    y = torch.randint(0, VOCAB, (3, 32))
    logits, loss, metrics = model(x, y, return_metrics=True)

    assert metrics["composite_loss"] is loss
    assert metrics["aux_loss"] > 0  # the tiny config has one MoE layer
    main = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    mtp = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), y[:, 1:].reshape(-1))
    assert torch.allclose(metrics["main_ce"], main)
    assert torch.allclose(metrics["mtp_ce"], mtp)
    recomposed = (metrics["main_ce"] + cfg.mtp_weight * metrics["mtp_ce"]
                  + cfg.moe_aux_weight * metrics["aux_loss"])
    assert torch.allclose(loss, recomposed, atol=1e-6)

    # the historical two-value API still works and returns the same loss
    _, loss_plain = model(x, y)
    assert torch.equal(loss_plain, loss)


def test_estimate_loss_reports_components_and_perplexity(tmp_path):
    cfg = make_cfg()
    torch.manual_seed(5)
    model = T.GPT(cfg)
    args = make_args(tmp_path / "ckpt.pt")
    data = make_data()
    result = T.estimate_loss(model, data, data, args, "cpu", contextlib.nullcontext())
    for split in ("train", "val"):
        values = result[split]
        for key in T.METRIC_KEYS:
            assert key in values
        assert values["main_perplexity"] == pytest.approx(math.exp(values["main_ce"]))
        recomposed = (values["main_ce"] + cfg.mtp_weight * values["mtp_ce"]
                      + cfg.moe_aux_weight * values["aux_loss"])
        # Means of per-batch components vs. the mean of per-batch composites
        # differ only by float summation order.
        assert values["composite_loss"] == pytest.approx(recomposed, rel=1e-6, abs=1e-6)


def test_fixed_eval_is_reproducible_and_keeps_the_training_rng(tmp_path):
    cfg = make_cfg()
    torch.manual_seed(5)
    model = T.GPT(cfg)
    args = make_args(tmp_path / "ckpt.pt")
    args.eval_mode, args.eval_seed = "fixed", 7
    data = make_data()

    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    first = T.estimate_loss(model, data, data, args, "cpu", contextlib.nullcontext())
    assert torch.equal(before, torch.get_rng_state()), "fixed evaluation touched the training RNG"

    torch.manual_seed(1234)  # a different global RNG state must not matter
    second = T.estimate_loss(model, data, data, args, "cpu", contextlib.nullcontext())
    assert first == second

    args.eval_seed = 8
    third = T.estimate_loss(model, data, data, args, "cpu", contextlib.nullcontext())
    assert third != first, "a different eval seed reused the same windows"


def test_legacy_eval_consumes_the_training_rng(tmp_path):
    """Historical behaviour, and the reason the mode must not change mid-run."""
    cfg = make_cfg()
    torch.manual_seed(5)
    model = T.GPT(cfg)
    args = make_args(tmp_path / "ckpt.pt")
    assert args.eval_mode == "legacy"
    data = make_data()
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    T.estimate_loss(model, data, data, args, "cpu", contextlib.nullcontext())
    assert not torch.equal(before, torch.get_rng_state())


def test_defaults_are_legacy_evaluation_and_composite_selection():
    args = T.parse_args([])
    assert args.eval_mode == "legacy"
    assert args.eval_seed == 12345
    assert args.select_metric == "composite"


# ------------------------------------------------------- selection semantics

def test_selection_metric_can_change_the_best_checkpoint(tmp_path, monkeypatch):
    """Two evaluations where main_ce and composite disagree on the better one."""
    scripted = [
        {"main_ce": 3.0, "mtp_ce": 5.0, "aux_loss": 1.0,
         "composite_loss": 3.0 + 0.5 * 5.0 + 0.01 * 1.0, "main_perplexity": math.exp(3.0)},
        {"main_ce": 3.1, "mtp_ce": 4.0, "aux_loss": 1.0,
         "composite_loss": 3.1 + 0.5 * 4.0 + 0.01 * 1.0, "main_perplexity": math.exp(3.1)},
    ]
    counter = {"i": 0}

    def fake_estimate(model, train_data, val_data, args, device, autocast):
        metrics = dict(scripted[counter["i"]])
        counter["i"] += 1
        return {"train": dict(metrics), "val": dict(metrics)}

    monkeypatch.setattr(T, "estimate_loss", fake_estimate)
    cfg, data = make_cfg(), make_data()

    def run(metric, name):
        counter["i"] = 0
        args = make_args(tmp_path / name / "ckpt.pt")
        args.max_iters, args.eval_interval, args.save_every = 3, 2, 0
        args.select_metric = metric
        torch.manual_seed(7)
        model = T.GPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        torch.manual_seed(99)
        stats = {}
        best_val, best_state, _ = T.run_training(
            model, opt, data, data, args, "cpu", contextlib.nullcontext(),
            {"type": "char"}, stats=stats)
        return best_val, best_state, stats["best_metrics"]

    composite_value, composite_state, composite_metrics = run("composite", "composite")
    main_value, main_state, main_metrics = run("main_ce", "main")

    # composite 5.51 vs 5.11 -> second evaluation; main_ce 3.0 vs 3.1 -> first
    assert composite_value == pytest.approx(5.11)
    assert composite_metrics["main_ce"] == pytest.approx(3.1)
    assert main_value == pytest.approx(3.0)
    assert main_metrics["mtp_ce"] == pytest.approx(5.0)
    assert any(not torch.equal(composite_state[k], main_state[k]) for k in composite_state)


def _trained_state(tmp_path, cfg, data, *, stop_after=3, **overrides):
    args = make_args(tmp_path / "ckpt.pt", stop_after=stop_after, **overrides)
    model, optimizer = fresh_model(cfg, args)
    torch.manual_seed(99)
    train_once(model, optimizer, args, data)
    return args, T.default_state_path(args.checkpoint)


def test_resume_rejects_changed_selection_metric(tmp_path):
    cfg, data = make_cfg(), make_data()
    _, path = _trained_state(tmp_path, cfg, data)
    changed = make_args(tmp_path / "ckpt.pt")
    changed.select_metric = "main_ce"
    with pytest.raises(ValueError) as err:
        T.load_train_state(path, cfg, changed, {"type": "char"})
    assert "metric" in str(err.value)


def test_resume_rejects_changed_eval_mode_and_seed(tmp_path):
    cfg, data = make_cfg(), make_data()
    _, path = _trained_state(tmp_path, cfg, data)  # legacy

    changed = make_args(tmp_path / "ckpt.pt")
    changed.eval_mode = "fixed"
    with pytest.raises(ValueError) as err:
        T.load_train_state(path, cfg, changed, {"type": "char"})
    assert "mode" in str(err.value)

    # legacy ignores the eval seed, so changing it alone is not a semantic change
    same = make_args(tmp_path / "ckpt.pt")
    same.eval_seed = 999
    T.load_train_state(path, cfg, same, {"type": "char"})

    _, fixed_path = _trained_state(tmp_path, cfg, data, eval_mode="fixed", eval_seed=7)
    other = make_args(tmp_path / "ckpt.pt")
    other.eval_mode, other.eval_seed = "fixed", 8
    with pytest.raises(ValueError) as err:
        T.load_train_state(fixed_path, cfg, other, {"type": "char"})
    assert "seed" in str(err.value)


def test_p01_state_without_evaluation_metadata_still_resumes(tmp_path):
    cfg, data = make_cfg(), make_data()
    _, path = _trained_state(tmp_path, cfg, data)

    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("evaluation", "selection", "best_metrics"):
        state.pop(key, None)
    for key in ("eval_mode", "eval_seed", "select_metric"):
        state["args"].pop(key, None)
    T.atomic_save(state, path)

    T.load_train_state(path, cfg, make_args(tmp_path / "ckpt.pt"), {"type": "char"})
    changed = make_args(tmp_path / "ckpt.pt")
    changed.select_metric = "main_ce"
    with pytest.raises(ValueError):
        T.load_train_state(path, cfg, changed, {"type": "char"})


def test_cli_records_evaluation_and_selection_metadata(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tinyshakespeare.txt").write_text(
        "ROMEO: first citizen: before we proceed any further, hear me speak.\n" * 400,
        encoding="utf-8")
    cmd = [
        sys.executable, os.path.join(V2, "train.py"), "--device", "cpu",
        "--tokenizer", "char", "--n_layer", "2", "--n_head", "2", "--n_kv_head", "1",
        "--n_embd", "32", "--block_size", "32", "--sliding_window", "16",
        "--n_experts", "4", "--n_experts_per_tok", "2", "--first_k_dense", "1",
        "--batch_size", "4", "--max_iters", "3", "--eval_interval", "1",
        "--eval_iters", "2", "--warmup_iters", "1", "--dropout", "0.0",
        "--save-every", "0", "--checkpoint", "ckpt.pt",
        "--eval-mode", "fixed", "--eval-seed", "3", "--select-metric", "main_ce",
    ]
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr
    assert "eval: fixed windows seed 3" in run.stdout
    assert "selection: main_ce" in run.stdout
    assert "comp | train main" in run.stdout and "ppl" in run.stdout

    ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["evaluation"]["mode"] == "fixed"
    assert ckpt["evaluation"]["seed"] == 3
    assert ckpt["selection"]["metric"] == "main_ce"
    assert ckpt["metrics"]["main_ce"] == pytest.approx(ckpt["selection"]["value"])
    assert ckpt["val_loss"] == pytest.approx(ckpt["metrics"]["composite_loss"])
    assert ckpt["metrics"]["main_perplexity"] == pytest.approx(
        math.exp(ckpt["metrics"]["main_ce"]))


# ------------------------------------------- device, config and prompt (P0-4)

def test_pick_device_auto_keeps_the_historical_pick(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert T.pick_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert T.pick_device("auto") == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert T.pick_device("auto") == "cpu"
    assert T.pick_device() == "cpu"


def test_pick_device_validates_explicit_choices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="--device cuda"):
        T.pick_device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert T.pick_device("cuda") == "cuda"
    # an explicit cpu never falls back to a visible GPU
    assert T.pick_device("cpu") == "cpu"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="--device mps"):
        T.pick_device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert T.pick_device("mps") == "mps"


def test_config_rejects_block_size_one_with_mtp():
    with pytest.raises(ValueError) as err:
        T.GPT(make_cfg(block_size=1, n_mtp=1))
    message = str(err.value)
    assert "block_size" in message and "n_mtp" in message


def test_config_allows_block_size_one_without_mtp():
    model = T.GPT(make_cfg(block_size=1, n_mtp=0)).eval()
    x = torch.randint(0, VOCAB, (2, 1))
    _, loss = model(x, x)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("over, needle", [
    ({"n_layer": 0}, "n_layer"),
    ({"n_embd": 0}, "n_embd"),
    ({"sliding_window": 0}, "sliding_window"),
    ({"full_attn_every": 0}, "full_attn_every"),
    ({"n_experts_per_tok": 5}, "n_experts_per_tok"),
    ({"n_mtp": -1}, "n_mtp"),
    ({"vocab_size": 0}, "vocab_size"),
])
def test_config_rejects_invalid_values(over, needle):
    with pytest.raises(ValueError) as err:
        T.GPT(make_cfg(**over))
    assert needle in str(err.value)


@pytest.mark.parametrize("over, needle", [
    ({"max_iters": 0}, "--max_iters"),
    ({"batch_size": 0}, "--batch_size"),
    ({"eval_iters": 0}, "--eval_iters"),
    ({"eval_interval": 0}, "--eval_interval"),
    ({"save_every": -1}, "--save-every"),
    ({"warmup_iters": -1}, "--warmup_iters"),
    ({"lr": 0.0}, "--lr"),
    ({"stop_after": -1}, "--stop-after"),
    ({"checkpoint": ""}, "--checkpoint"),
    ({"block_size": 1}, "block_size"),
])
def test_validate_args_rejects_invalid_values(tmp_path, over, needle):
    args = make_args(tmp_path / "ckpt.pt")
    for key, value in over.items():
        setattr(args, key, value)
    with pytest.raises(ValueError) as err:
        T.validate_args(args)
    assert needle in str(err.value)


def test_validate_args_accepts_the_defaults(tmp_path):
    T.validate_args(make_args(tmp_path / "ckpt.pt"))
    T.validate_args(T.parse_args([]))


def _char_tokenizer(tmp_path, monkeypatch):
    corpus = tmp_path / "mini.txt"
    corpus.write_text("abc abc abc\n" * 50, encoding="utf-8")
    monkeypatch.setattr(T, "DATA_PATH", str(corpus))
    _, _, encode, decode, vocab, _ = T.load_data("char", 1000)
    return encode, decode, vocab


def test_generate_rejects_characters_outside_the_vocabulary(tmp_path, monkeypatch):
    encode, decode, vocab = _char_tokenizer(tmp_path, monkeypatch)
    model = T.GPT(make_cfg(vocab_size=vocab, block_size=8)).eval()
    with pytest.raises(ValueError) as err:
        T.generate(model, encode, decode, "cpu", prompt="abZ", max_new_tokens=1)
    message = str(err.value)
    assert "'Z'" in message and "vocabulary" in message and "abZ" in message


def test_generate_accepts_a_known_char_prompt(tmp_path, monkeypatch):
    encode, decode, vocab = _char_tokenizer(tmp_path, monkeypatch)
    model = T.GPT(make_cfg(vocab_size=vocab, block_size=8)).eval()
    text = T.generate(model, encode, decode, "cpu", prompt="abc", max_new_tokens=4)
    assert isinstance(text, str) and text.startswith("abc")


def test_generate_validates_sampling_parameters(tmp_path, monkeypatch):
    encode, decode, vocab = _char_tokenizer(tmp_path, monkeypatch)
    model = T.GPT(make_cfg(vocab_size=vocab, block_size=8)).eval()
    with pytest.raises(ValueError, match="prompt"):
        T.generate(model, encode, decode, "cpu", prompt="")
    with pytest.raises(ValueError, match="temperature"):
        T.generate(model, encode, decode, "cpu", prompt="abc", temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        T.generate(model, encode, decode, "cpu", prompt="abc", top_k=0)


def test_bpe_prompt_accepts_characters_outside_the_training_text():
    pytest.importorskip("sentencepiece")
    model_path = os.path.join(V2, "data", "tinyshakespeare_bpe1000.model")
    if not os.path.exists(model_path):
        pytest.skip("no BPE tokenizer in this worktree")
    encode, decode = T.tokenizer_from_info({"type": "bpe", "model_path": model_path})
    prompt = "Ω≈ç√✅"
    ids = encode(prompt)
    assert ids, "byte fallback must encode any character"
    assert decode(ids) == prompt
    model = T.GPT(make_cfg(vocab_size=1000, block_size=8)).eval()
    text = T.generate(model, encode, decode, "cpu", prompt=prompt, max_new_tokens=2)
    assert isinstance(text, str) and text


def _tiny_cli(tmp_path, corpus, extra=None, device="cpu"):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "tinyshakespeare.txt").write_text(corpus, encoding="utf-8")
    cmd = [
        sys.executable, os.path.join(V2, "train.py"), "--device", device,
        "--tokenizer", "char", "--n_layer", "1", "--n_head", "2", "--n_kv_head", "1",
        "--n_embd", "32", "--block_size", "16", "--sliding_window", "8",
        "--n_experts", "4", "--n_experts_per_tok", "2", "--first_k_dense", "1",
        "--batch_size", "4", "--max_iters", "2", "--eval_interval", "1",
        "--eval_iters", "1", "--warmup_iters", "1", "--dropout", "0.0",
        "--save-every", "0", "--checkpoint", "ckpt.pt",
    ]
    return cmd + (extra or [])


def test_cli_rejects_invalid_eval_iters(tmp_path):
    cmd = _tiny_cli(tmp_path, "abc\n" * 100, ["--eval_iters", "0"])
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert run.returncode == 1
    assert "--eval_iters" in run.stderr
    assert "Traceback" not in run.stderr


def test_cli_rejects_an_unavailable_device(tmp_path):
    cmd = _tiny_cli(tmp_path, "abc\n" * 100, device="cuda")
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert run.returncode == 1
    assert "--device cuda" in run.stderr
    assert "Traceback" not in run.stderr


def test_cli_skips_a_sample_it_cannot_encode(tmp_path):
    cmd = _tiny_cli(tmp_path, "the quick brown fox jumps over the lazy dog. " * 300)
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr
    assert "sample generation skipped" in run.stdout
    assert "'R'" in run.stdout
    assert (tmp_path / "ckpt.pt").exists()
