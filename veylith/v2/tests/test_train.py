"""CPU-only tests for checkpointing, resume and the loss wiring.

Run from the repository root:

    python -m pytest veylith/v2/tests -q

Everything here is forced onto the CPU (``--device cpu`` or ``pick_device``
called directly); no test touches a GPU.
"""

import contextlib
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
