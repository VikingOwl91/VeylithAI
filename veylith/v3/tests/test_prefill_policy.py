"""CPU tests for the deterministic-prefill runtime policy (PERF-4D).

The policy is a process-global runtime setting, never a config field, so these
tests also pin that it does not enter the configuration fingerprint.
"""

import pytest

torch = pytest.importorskip("torch")

from architecture.config import tiny_config  # noqa: E402
from training.config import TrainConfig  # noqa: E402
from training.determinism import (PREFILL_ENV, apply_prefill_policy,  # noqa: E402
                                  current_prefill_state,
                                  describe_prefill_policy, parse_prefill_policy,
                                  resolve_prefill_policy, restore_prefill_state,
                                  restores_prefill_state)


@pytest.fixture(autouse=True)
def restore_global_state():
    previous = torch.utils.deterministic.fill_uninitialized_memory
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    restore_prefill_state(previous)
    torch.use_deterministic_algorithms(deterministic)


def test_default_training_mode_disables_prefill(monkeypatch):
    monkeypatch.delenv(PREFILL_ENV, raising=False)
    policy = resolve_prefill_policy()
    assert policy == {"enabled": False, "env_value": None, "source": "default"}
    apply_prefill_policy(policy)
    assert current_prefill_state() is False


def test_debug_override_enables_prefill(monkeypatch):
    monkeypatch.setenv(PREFILL_ENV, "1")
    policy = resolve_prefill_policy()
    assert policy["enabled"] is True
    assert policy["source"] == "environment"
    apply_prefill_policy(policy)
    assert current_prefill_state() is True


@pytest.mark.parametrize("value", ["0", ""])
def test_explicit_disable_values(value):
    assert parse_prefill_policy(value) is False


def test_unset_is_disabled_not_enabled():
    assert parse_prefill_policy(None) is False


@pytest.mark.parametrize("value", ["true", "TRUE", "yes", "on", "2", "-1",
                                   " 1", "1 ", "0.0", "enabled"])
def test_invalid_values_are_rejected(value):
    with pytest.raises(ValueError, match=PREFILL_ENV):
        parse_prefill_policy(value)


def test_invalid_environment_value_fails_resolution(monkeypatch):
    monkeypatch.setenv(PREFILL_ENV, "enabled")
    with pytest.raises(ValueError, match=PREFILL_ENV):
        resolve_prefill_policy()


def test_deterministic_algorithms_remain_enabled():
    torch.use_deterministic_algorithms(True, warn_only=False)
    apply_prefill_policy({"enabled": False})
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    assert current_prefill_state() is False


def test_state_restored_after_normal_exit():
    restore_prefill_state(True)

    @restores_prefill_state
    def run():
        apply_prefill_policy({"enabled": False})
        assert current_prefill_state() is False
        return "done"

    assert run() == "done"
    assert current_prefill_state() is True


def test_state_restored_after_exception():
    restore_prefill_state(True)

    @restores_prefill_state
    def run():
        apply_prefill_policy({"enabled": False})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run()
    assert current_prefill_state() is True


def test_state_restored_when_policy_application_is_skipped():
    restore_prefill_state(False)

    @restores_prefill_state
    def run():
        raise ValueError(f"{PREFILL_ENV} must be '1'")

    with pytest.raises(ValueError):
        run()
    assert current_prefill_state() is False


def test_startup_log_line_reports_effective_policy(monkeypatch):
    monkeypatch.delenv(PREFILL_ENV, raising=False)
    assert describe_prefill_policy(resolve_prefill_policy()) == (
        f"deterministic prefill disabled ({PREFILL_ENV} unset)")
    monkeypatch.setenv(PREFILL_ENV, "1")
    line = describe_prefill_policy(resolve_prefill_policy())
    assert "enabled" in line and "='1'" in line


def _config():
    return TrainConfig(
        data_root="data", out_dir="out",
        model=tiny_config(vocab_size=512, block_size=16, n_layer=1, n_head=1,
                          n_kv_head=1, n_embd=16, n_experts=2,
                          n_experts_per_tok=1, expert_ffn_dim=16,
                          shared_expert=False, first_k_dense=1),
        device="cpu", precision="fp32", batch_size=1, grad_accum=1,
        seq_len=16, max_steps=2)


def test_policy_does_not_change_the_configuration_fingerprint(monkeypatch):
    cfg = _config()
    monkeypatch.delenv(PREFILL_ENV, raising=False)
    resolve_prefill_policy()
    disabled = cfg.fingerprint("0" * 64)
    monkeypatch.setenv(PREFILL_ENV, "1")
    resolve_prefill_policy()
    enabled = cfg.fingerprint("0" * 64)
    assert disabled == enabled
    assert PREFILL_ENV not in repr(cfg.to_dict())
