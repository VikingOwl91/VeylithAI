#!/usr/bin/env python3
"""Read-only drift guard for the isolated Veylith training environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env_fingerprint import (EXPECTED_TORCH_SHA256, EXPECTED_TORCH_VERSION,
                             collect)


def _issue(issues, severity, name, expected, current):
    issues.append({
        "severity": severity,
        "name": name,
        "expected": expected,
        "current": current,
    })


def compare(expected: dict, current: dict, strict_host: bool) -> list[dict]:
    issues = []
    if expected.get("schema_version") != 1 or current.get("schema_version") != 1:
        _issue(issues, "critical", "schema_version",
               expected.get("schema_version"), current.get("schema_version"))
        return issues

    for key in ("version", "executable", "prefix", "base_prefix"):
        if expected["python"].get(key) != current["python"].get(key):
            _issue(issues, "critical", f"python.{key}",
                   expected["python"].get(key), current["python"].get(key))
    if current["python"].get("user_site"):
        _issue(issues, "critical", "python.user_site", False, True)
    if current["python"].get("site_packages_leak"):
        _issue(issues, "critical", "python.site_packages_leak", [],
               current["python"]["site_packages_leak"])

    expected_packages = expected.get("packages", {})
    current_packages = current.get("packages", {})
    for name in sorted(set(expected_packages) | set(current_packages)):
        if expected_packages.get(name) != current_packages.get(name):
            _issue(issues, "critical", f"package.{name}",
                   expected_packages.get(name), current_packages.get(name))

    if expected.get("lock", {}).get("sha256") != current.get("lock", {}).get("sha256"):
        _issue(issues, "critical", "lock.sha256",
               expected.get("lock", {}).get("sha256"),
               current.get("lock", {}).get("sha256"))
    expected_uv = expected.get("lock", {}).get("uv_lock") or {}
    current_uv = current.get("lock", {}).get("uv_lock") or {}
    if expected_uv.get("sha256") != current_uv.get("sha256"):
        _issue(issues, "critical", "lock.uv_lock",
               expected_uv.get("sha256"), current_uv.get("sha256"))
    torch_hashes = current.get("lock", {}).get("torch_hashes", [])
    if EXPECTED_TORCH_SHA256 not in torch_hashes:
        _issue(issues, "critical", "lock.torch_sha256",
               EXPECTED_TORCH_SHA256, torch_hashes)

    for key in ("version", "hip", "config_sha256"):
        if expected["torch"].get(key) != current["torch"].get(key):
            _issue(issues, "critical", f"torch.{key}",
                   expected["torch"].get(key), current["torch"].get(key))
    if current["torch"].get("version") != EXPECTED_TORCH_VERSION:
        _issue(issues, "critical", "torch.expected_version",
               EXPECTED_TORCH_VERSION, current["torch"].get("version"))

    native = current.get("native", {})
    if native.get("opt_rocm_lib_count") != 0:
        _issue(issues, "critical", "native.opt_rocm_lib_count", 0,
               native.get("opt_rocm_lib_count"))
    maps = current.get("maps", {})
    if maps.get("opt_rocm_count") != 0:
        _issue(issues, "critical", "maps.opt_rocm_count", 0,
               maps.get("opt_rocm_count"))
    rpath = native.get("rpath", []) + native.get("runpath", [])
    if "$ORIGIN" not in rpath:
        _issue(issues, "critical", "native.rpath", ["$ORIGIN"], rpath)

    host_severity = "critical" if strict_host else "warning"
    expected_host = expected.get("host", {})
    current_host = current.get("host", {})
    for key in ("kernel", "rocm_system", "packages"):
        if expected_host.get(key) != current_host.get(key):
            _issue(issues, host_severity, f"host.{key}",
                   expected_host.get(key), current_host.get(key))

    expected_gpu = expected.get("gpu", [])
    current_gpu = current.get("gpu", [])
    if expected_gpu != current_gpu:
        _issue(issues, "critical", "gpu", expected_gpu, current_gpu)

    if expected.get("uv", {}).get("version") != current.get("uv", {}).get("version"):
        _issue(issues, "warning", "uv.version",
               expected.get("uv", {}).get("version"),
               current.get("uv", {}).get("version"))
    return issues


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected", default=str(repo_root / "veylith/v3/env/env.baseline.json"))
    parser.add_argument(
        "--lock", default=str(repo_root / "veylith/v3/env/requirements.lock"))
    parser.add_argument("--strict-host", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    current = collect(args.lock, repo_root)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    issues = compare(expected, current, args.strict_host)
    critical = [issue for issue in issues if issue["severity"] == "critical"]
    for issue in issues:
        print(f"{issue['severity'].upper()} {issue['name']}: "
              f"expected={issue['expected']!r} current={issue['current']!r}")
    if not issues:
        print("OK: isolated environment matches the accepted baseline")
    elif critical:
        print(f"FAIL: {len(critical)} critical drift issue(s)")
    else:
        print("WARN: environment matches critical fields; host drift is recorded")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
