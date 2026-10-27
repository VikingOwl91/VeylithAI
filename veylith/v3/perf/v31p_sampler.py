"""Read-only telemetry sampler for v3.1-PERF-1.

Samples GPU power (PPT), clocks, temperature, busy percent, KFD PIDs and system
CPU utilisation at a fixed interval into a CSV. It changes no settings.

Run with the system interpreter (psutil is available there); the isolated
environment is left untouched.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

CARD = "/sys/class/drm/card1/device"
HWMON = os.path.join(CARD, "hwmon", "hwmon3")


def read_int(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return int(handle.read().strip())
    except Exception:
        return None


def kfd_pids():
    pids = []
    for path in glob.glob("/sys/class/kfd/kfd/proc/*"):
        name = os.path.basename(path)
        if name.isdigit():
            pids.append(int(name))
    return sorted(pids)


def cpu_times():
    with open("/proc/stat", encoding="utf-8") as handle:
        fields = handle.readline().split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=880.0)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    started = time.monotonic()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["monotonic", "wall_utc", "power_uw", "sclk_hz",
                         "mclk_hz", "temp_millic", "busy_pct", "kfd_pids",
                         "sys_idle", "sys_total"])
        while time.monotonic() - started < args.duration:
            idle, total = cpu_times()
            writer.writerow([
                f"{time.monotonic():.6f}",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                read_int(os.path.join(HWMON, "power1_average")),
                read_int(os.path.join(HWMON, "freq1_input")),
                read_int(os.path.join(HWMON, "freq2_input")),
                read_int(os.path.join(HWMON, "temp1_input")),
                read_int(os.path.join(CARD, "gpu_busy_percent")),
                " ".join(str(pid) for pid in kfd_pids()),
                idle,
                total,
            ])
            handle.flush()
            time.sleep(args.interval)
    print("sampler wrote", args.out)


if __name__ == "__main__":
    main()
