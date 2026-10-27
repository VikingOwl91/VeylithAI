#!/usr/bin/env python3
"""Collect a read-only fingerprint of the isolated Veylith training stack."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import site
import stat
import subprocess
import sys
from importlib import metadata
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

EXPECTED_TORCH_VERSION = "2.14.1+rocm7.2"
EXPECTED_TORCH_SHA256 = (
    "e378baf90ba06c988e234449787a6fd9dac60dcd869eef35cfcc9546cd3f1f14"
)
EXPECTED_TORCH_WHEEL = (
    "torch-2.14.1+rocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_packages() -> dict[str, str]:
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[normalize(name)] = dist.version
    return dict(sorted(packages.items()))


def python_info() -> dict:
    venv_site = (Path(sys.prefix) / "lib" /
                 f"python{sys.version_info.major}.{sys.version_info.minor}" /
                 "site-packages")
    leaks = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        if path == venv_site:
            continue
        if path.name in ("site-packages", "dist-packages"):
            leaks.append(str(path))
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "user_site": bool(site.ENABLE_USER_SITE),
        "user_site_path": site.getusersitepackages(),
        "sys_path": list(sys.path),
        "site_packages_leak": leaks,
        "platform": sys.platform,
    }


def uv_info() -> dict:
    output = run_readonly(["uv", "--version"])
    return {"version": output.strip() if output else None}


def run_readonly(command: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def torch_info() -> dict:
    import torch

    config = torch.__config__.show()
    return {
        "version": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "file": torch.__file__,
        "config": config,
        "config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
    }


def native_info(torch_file: str) -> dict:
    torch_lib = Path(torch_file).resolve().parent / "lib"
    candidates = [torch_lib / "libtorch_hip.so",
                  torch_lib / "libtorch_hip.so.2.14.1"]
    libtorch_hip = next((path for path in candidates if path.exists()), None)
    result = {
        "libtorch_hip": str(libtorch_hip) if libtorch_hip else None,
        "rpath": [],
        "runpath": [],
        "ldd": [],
        "opt_rocm_lib_count": 0,
        "rocm_like": [],
        "host_like": [],
    }
    if libtorch_hip is None:
        return result

    readelf = run_readonly(["readelf", "-d", str(libtorch_hip)])
    if readelf:
        for line in readelf.splitlines():
            if "RPATH" in line or "RUNPATH" in line:
                match = re.search(r"\[(.*?)\]", line)
                if match:
                    values = [item for item in match.group(1).split(":") if item]
                    if "RPATH" in line:
                        result["rpath"] = values
                    else:
                        result["runpath"] = values

    ldd = run_readonly(["ldd", str(libtorch_hip)])
    if ldd:
        for line in ldd.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\S+)\s+=>\s+(\S+)", line)
            if match:
                name, path = match.group(1), match.group(2)
                found = path != "not"
            else:
                match = re.match(r"^(\S+)\s+\(0x", line)
                if not match:
                    continue
                name, path, found = match.group(1), None, True
            entry = {"name": name, "path": path, "found": found}
            result["ldd"].append(entry)
            if path and path.startswith("/opt/rocm"):
                result["opt_rocm_lib_count"] += 1
            text = f"{name} {path or ''}".lower()
            if any(token in text for token in
                   ("rocm", "hip", "amd", "miopen", "rccl", "hsa", "comgr")):
                result["rocm_like"].append(entry)
            if any(token in name for token in
                   ("libc.so", "libm.so", "libstdc++", "libgcc_s",
                    "libzstd", "liblzma", "libnuma", "libgomp",
                    "libpthread", "librt", "libdl")):
                result["host_like"].append(entry)
    return result


def maps_info() -> dict:
    paths = set()
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(None, 5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    paths.add(fields[5])
    except OSError:
        pass
    relevant = sorted(path for path in paths if any(
        token in path.lower() for token in
        ("/torch/lib/", "/opt/rocm", "rocm", "hip", "amd", "miopen", "rccl", "hsa")))
    opt_rocm = sorted(path for path in paths if path.startswith("/opt/rocm"))
    return {
        "relevant": relevant,
        "opt_rocm": opt_rocm,
        "opt_rocm_count": len(opt_rocm),
    }


def host_info() -> dict:
    packages = {}
    output = run_readonly([
        "pacman", "-Q", "glibc", "gcc-libs", "zstd", "xz", "linux-zen",
        "linux-firmware-amdgpu", "amdgpu_top",
    ])
    if output:
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 2:
                packages[fields[0]] = fields[1]
    rocm_version = None
    rocm_file = Path("/opt/rocm/.info/version")
    if rocm_file.exists():
        rocm_version = rocm_file.read_text(encoding="utf-8").strip()
    devices = []
    for path in ("/dev/kfd", "/dev/dri/renderD128", "/dev/dri/card1"):
        try:
            info = os.stat(path)
        except OSError:
            devices.append({"path": path, "exists": False})
            continue
        devices.append({
            "path": path,
            "exists": True,
            "mode": stat.filemode(info.st_mode),
            "uid": info.st_uid,
            "gid": info.st_gid,
        })
    return {
        "kernel": platform.release(),
        "rocm_system": rocm_version,
        "packages": dict(sorted(packages.items())),
        "devices": devices,
    }


def gpu_info() -> list[dict]:
    devices = []
    for uevent in sorted(Path("/sys/class/drm").glob("card*/device/uevent")):
        data = {}
        try:
            for line in uevent.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    data[key] = value
        except OSError:
            continue
        if data.get("DRIVER") != "amdgpu":
            continue
        vendor = uevent.parent / "vendor"
        device = uevent.parent / "device"
        devices.append({
            "card": uevent.parent.parent.name,
            "driver": data.get("DRIVER"),
            "pci_id": data.get("PCI_ID"),
            "pci_subsys_id": data.get("PCI_SUBSYS_ID"),
            "pci_class": data.get("PCI_CLASS"),
            "vendor": vendor.read_text(encoding="utf-8").strip()
            if vendor.exists() else None,
            "device": device.read_text(encoding="utf-8").strip()
            if device.exists() else None,
        })
    return devices


def lock_info(path: str | Path) -> dict:
    packages = {}
    hashes: dict[str, list[str]] = {}
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().rstrip("\\").strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                if current and stripped.startswith("--hash=sha256:"):
                    hashes.setdefault(current, []).append(
                        stripped.split("sha256:", 1)[1])
                continue
            if "==" in stripped:
                name, version = stripped.split("==", 1)
                name = normalize(name)
                version = version.split(";", 1)[0].strip()
                packages[name] = version
                current = name
    uv_lock = Path(path).with_name("uv.lock")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "uv_lock": {
            "path": str(uv_lock),
            "sha256": sha256_file(uv_lock),
        } if uv_lock.exists() else None,
        "packages": dict(sorted(packages.items())),
        "torch_hashes": sorted(hashes.get("torch", [])),
    }


def repo_head(repo_root: Path) -> str | None:
    return run_readonly(["git", "-C", str(repo_root), "rev-parse", "HEAD"])


def collect(lock_path: str | Path, repo_root: Path | None = None) -> dict:
    lock_path = Path(lock_path)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    torch = torch_info()
    native = native_info(torch["file"])
    maps = maps_info()
    return {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_head": repo_head(repo_root),
        "python": python_info(),
        "uv": uv_info(),
        "packages": installed_packages(),
        "torch": torch,
        "native": native,
        "maps": maps,
        "host": host_info(),
        "gpu": gpu_info(),
        "lock": lock_info(lock_path),
        "expected": {
            "torch_version": EXPECTED_TORCH_VERSION,
            "torch_sha256": EXPECTED_TORCH_SHA256,
            "torch_wheel": EXPECTED_TORCH_WHEEL,
        },
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", default=str(repo_root / "veylith/v3/env/requirements.lock"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    fingerprint = collect(args.lock, repo_root)
    payload = json.dumps(fingerprint, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(out)
        print(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
