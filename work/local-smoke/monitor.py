"""Record resource use for one HoloBoxing smoke-test subprocess."""

import csv
from datetime import datetime
import json
from pathlib import Path
import resource
import subprocess
import sys
from threading import Event, Thread
import time
from zoneinfo import ZoneInfo


def sample():
    gpu = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits", "-i", "0",
        ],
        capture_output=True, text=True, timeout=4, check=True,
    )
    used, total = (int(part.strip()) for part in gpu.stdout.strip().split(","))
    memory = {}
    with open("/proc/meminfo", encoding="ascii") as counters:
        for line in counters:
            key, value = line.split(":", 1)
            if key in ("MemAvailable", "SwapFree"):
                memory[key] = round(int(value.split()[0]) / 1024, 1)
    return used, total, memory["MemAvailable"], memory["SwapFree"]


def oom_kills():
    for line in Path("/proc/vmstat").read_text(encoding="ascii").splitlines():
        if line.startswith("oom_kill "):
            return int(line.split()[1])
    return None


def run(label, cmd):
    project = Path(__file__).resolve().parents[2]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    folder = project / "outputs" / "local-smoke" / "logs" / f"{stamp}-{label}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    oom_before = oom_kills()
    initial = sample()
    samples = [(0, *initial)]
    stop = Event()

    def record():
        while not stop.wait(0.65):
            try:
                values = sample()
            except (subprocess.SubprocessError, ValueError, OSError):
                continue
            samples.append((round(time.monotonic() - started, 3), *values))

    worker = Thread(target=record, daemon=True)
    print("LOG_DIRECTORY", folder, flush=True)
    worker.start()
    with (folder / "command.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            cmd, cwd=project / "upstream" / "HoloMotion",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            result = proc.wait()
    stop.set()
    worker.join(timeout=5)
    with (folder / "resources.csv").open("w", newline="", encoding="ascii") as log:
        writer = csv.writer(log)
        writer.writerow(("seconds", "gpu_used_mib", "gpu_total_mib", "wsl_available_mib", "wsl_swap_free_mib"))
        writer.writerows(samples)
    report = {
        "stage": label,
        "command": cmd,
        "exit_code": result,
        "seconds": round(time.monotonic() - started, 2),
        "samples": len(samples),
        "gpu_total_mib": initial[1],
        "gpu_baseline_used_mib": initial[0],
        "gpu_peak_card_used_mib": max(row[1] for row in samples),
        "wsl_min_available_mib": min(row[3] for row in samples),
        "wsl_min_swap_free_mib": min(row[4] for row in samples),
        "child_max_rss_mib": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
    }
    oom_after = oom_kills()
    if oom_before is not None and oom_after is not None:
        report["wsl_oom_kills_during_stage"] = oom_after - oom_before
    (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("STAGE_RESULT", json.dumps(report), flush=True)
    return result if result >= 0 else 128 - result


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2:]))
