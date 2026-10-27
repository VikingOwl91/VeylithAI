"""CPU-only checks for the isolated-environment drift guard."""

import copy
import json
import sys
from pathlib import Path

import pytest

ENV_DIR = Path(__file__).resolve().parents[1] / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from env_check import compare  # noqa: E402


@pytest.fixture
def baseline():
    return json.loads((ENV_DIR / "env.baseline.json").read_text(encoding="utf-8"))


def test_guard_accepts_the_baseline(baseline):
    assert compare(baseline, copy.deepcopy(baseline), strict_host=False) == []


def test_guard_rejects_package_drift(baseline):
    current = copy.deepcopy(baseline)
    current["packages"]["numpy"] = "0.0.0"
    issues = compare(baseline, current, strict_host=False)
    assert any(issue["name"] == "package.numpy" and
               issue["severity"] == "critical" for issue in issues)


def test_guard_rejects_site_package_leakage(baseline):
    current = copy.deepcopy(baseline)
    current["python"]["site_packages_leak"] = ["/usr/lib/python3.14/site-packages"]
    issues = compare(baseline, current, strict_host=False)
    assert any(issue["name"] == "python.site_packages_leak" and
               issue["severity"] == "critical" for issue in issues)


def test_guard_rejects_opt_rocm_leakage(baseline):
    current = copy.deepcopy(baseline)
    current["native"]["opt_rocm_lib_count"] = 1
    issues = compare(baseline, current, strict_host=False)
    assert any(issue["name"] == "native.opt_rocm_lib_count" and
               issue["severity"] == "critical" for issue in issues)


def test_guard_rejects_changed_lock(baseline):
    current = copy.deepcopy(baseline)
    current["lock"]["sha256"] = "0" * 64
    issues = compare(baseline, current, strict_host=False)
    assert any(issue["name"] == "lock.sha256" and
               issue["severity"] == "critical" for issue in issues)


def test_guard_rejects_changed_uv_lock(baseline):
    current = copy.deepcopy(baseline)
    current["lock"]["uv_lock"]["sha256"] = "0" * 64
    issues = compare(baseline, current, strict_host=False)
    assert any(issue["name"] == "lock.uv_lock" and
               issue["severity"] == "critical" for issue in issues)


def test_guard_host_drift_warns_and_strict_host_fails(baseline):
    current = copy.deepcopy(baseline)
    current["host"]["kernel"] = "0.0.0-test"
    warned = compare(baseline, current, strict_host=False)
    strict = compare(baseline, current, strict_host=True)
    assert any(issue["name"] == "host.kernel" and
               issue["severity"] == "warning" for issue in warned)
    assert any(issue["name"] == "host.kernel" and
               issue["severity"] == "critical" for issue in strict)
