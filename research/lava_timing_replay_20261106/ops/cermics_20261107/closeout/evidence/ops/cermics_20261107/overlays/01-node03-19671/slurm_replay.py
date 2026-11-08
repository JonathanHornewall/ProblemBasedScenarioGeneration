"""Submit and run the six pinned routes with a durable SP25 file gate."""
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/jonathan.hornewall/runs/lava_timing_replay_20261107")
OPS = ROOT / "ops/cermics_20261107"
SCRATCH = Path("/tmp/lava_timing_replay_20261107")
OWNER = "jonathan.hornewall"
GIB = 2**30


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def cutoff(plan):
    return dt.datetime.fromisoformat(plan["cutoff_utc"].replace("Z", "+00:00")).timestamp()


def load_plan():
    plan = read(OPS / "RUN_PLAN.json")
    manifest_path = ROOT / "manifest.json"
    manifest = read(manifest_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_routes = {"sp25", "sp10", "sp15", "sp20", "cfl50", "kp1000"}
    if (
        plan["remote_root"] != str(ROOT)
        or plan["scratch"] != str(SCRATCH)
        or plan["cutoff_utc"] != "2026-11-09T22:30:00Z"
        or plan["manifest_sha256"] != manifest_hash
        or set(plan["routes"]) != expected_routes
        or len(manifest["jobs"]) != 19
        or {item["key"] for item in plan["routes"].values()} != set(manifest["banks"])
        or manifest["jobs"][manifest["first"]]["job"]["required_data_key"] != plan["routes"]["sp25"]["key"]
    ):
        raise RuntimeError("Frozen execution plan or manifest differs")
    return plan, manifest


def reconcile(intent):
    """Look up one exact owned submission. Never turn absence into a new submit."""
    name = intent["name"]
    queries = [
        ["squeue", "-h", "-u", OWNER, "-n", name, "-o", "%i|%j|%T"],
        ["sacct", "-X", "-n", "-P", "-u", OWNER, "--name", name,
         "--starttime", intent["started_utc"][:10], "-o", "JobIDRaw,JobName%96,State%36"],
    ]
    matches = {}
    for args in queries:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=12)
        if proc.returncode:
            continue
        for line in proc.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 3 and fields[0].isdigit() and fields[1] == name:
                matches[fields[0]] = fields[2]
    if len(matches) != 1:
        raise RuntimeError(f"{name} has {len(matches)} exact matches. Reconcile before retrying.")
    job_id, state = next(iter(matches.items()))
    return {"job_id": job_id, "name": name, "state": state,
            "reconciled": True, "recorded_utc": utc_now()}


def submit_route(plan, label):
    if STOP_REQUESTED or (ROOT / "STOP").exists() or (ROOT / "TERMINAL.json").exists():
        raise RuntimeError("Campaign admissions are closed")
    route = plan["routes"][label]
    folder = OPS / "submissions" / label
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "RECEIPT.json"
    intent_path = folder / "INTENT.json"
    with (folder / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            return read(receipt_path)
        if intent_path.exists():
            receipt = reconcile(read(intent_path))
            write(receipt_path, receipt)
            return receipt
        minutes = min(route["wall_minutes"], int((cutoff(plan) - time.time() - 60) // 60))
        if minutes < 5:
            raise RuntimeError("Less than five minutes remain before the fixed cutoff")
        name = f"lava-replay-1107-{label}"
        argv = [
            "sbatch", "--parsable", f"--job-name={name}", f"--partition={route['partition']}",
            f"--nodelist={route['node']}", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
            f"--mem={route['memory_GiB']}G", "--nice=10000", "--requeue",
            f"--time={minutes}", f"--deadline={plan['cutoff_utc'].removesuffix('Z')}",
            "--open-mode=append", f"--chdir={ROOT}",
            f"--output={ROOT}/logs/%x-%j.out", f"--error={ROOT}/logs/%x-%j.err",
            str(OPS / "worker.sbatch"), label,
        ]
        intent = {"label": label, "name": name, "started_utc": utc_now(), "argv": argv,
                  "manifest_sha256": plan["manifest_sha256"], "cutoff_utc": plan["cutoff_utc"]}
        write(intent_path, intent)
        try:
            proc = subprocess.run(
                argv, env={**os.environ, "TZ": "UTC"}, capture_output=True, text=True, timeout=20
            )
            job_id = proc.stdout.strip().split(";")[0]
            if proc.returncode or not job_id.isdigit():
                raise RuntimeError(f"sbatch status {proc.returncode}: {proc.stderr.strip()[:1000]}")
            receipt = {"job_id": job_id, "name": name, "recorded_utc": utc_now(), "reconciled": False}
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            write(folder / "SUBMISSION_ERROR.json", {"error": str(error), "recorded_utc": utc_now()})
            receipt = reconcile(intent)
        write(receipt_path, receipt)
        return receipt


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is unavailable")


def process_tree(pid):
    """Sum live RSS and CPU counters for this exact child and its descendants."""
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        cpu = sum(int(fields[i]) for i in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
        rss = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        children = (path / "task" / str(pid) / "children").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return 0.0, 0
    for child in children:
        child_cpu, child_rss = process_tree(int(child))
        cpu += child_cpu
        rss += child_rss
    return cpu, rss


def run_phase(plan, label, phase, attempt):
    route = plan["routes"][label]
    limit = (plan["first_phase_RSS_guard_GiB"] if phase == "first" else route["memory_GiB"]) * GIB
    headroom = plan["host_headroom_GiB"] * GIB
    scratch_reserve = plan["scratch_running_reserve_GiB"] * GIB
    start = time.time()
    status = {"route": label, "phase": phase, "started_unix": start,
              "host": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
              "step_id": os.environ.get("SLURM_STEP_ID"), "affinity": sorted(os.sched_getaffinity(0)),
              "memory_limit_bytes": limit, "state": "admission", "allocated_CPUs": 1,
              "partition": route["partition"], "MemAvailable_before_bytes": available_memory(),
              "scratch_free_before_bytes": shutil.disk_usage("/tmp").free}
    path = attempt / (phase + ".json")
    write(path, status)
    try:
        if status["MemAvailable_before_bytes"] < limit + headroom:
            raise RuntimeError("Insufficient MemAvailable for route allowance plus 8 GiB host headroom")
        if status["scratch_free_before_bytes"] < route["scratch_admission_GiB"] * GIB:
            raise RuntimeError("Insufficient actual node-local scratch before starting route")
        if STOP_REQUESTED or (ROOT / "STOP").exists() or time.time() >= cutoff(plan) - 30:
            raise RuntimeError("Fixed scientific cutoff reached")
        filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", "/tmp"], text=True).strip()
        if filesystem in {"nfs", "nfs4", "cifs", "smb2"} or SCRATCH.is_symlink():
            raise RuntimeError("Expected owned node-local scratch")
        if SCRATCH.exists() and SCRATCH.stat().st_uid != os.getuid():
            raise RuntimeError("Scratch exists under a different owner")
        status["scratch_filesystem"] = filesystem
        argv = ["bash", str(ROOT / "start_cermics.sh"), route["key"], phase]
        with (attempt / (phase + ".log")).open("xb") as stream:
            proc = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(state="running", pid=proc.pid, argv=argv)
            write(path, status)
            peak = 0
            last_write = 0.0
            last_sample = None
            stop_reason = None
            kill_at = None
            while True:
                pid, wait_status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    proc.returncode = os.waitstatus_to_exitcode(wait_status)
                    break
                now = time.time()
                cpu, rss = process_tree(proc.pid)
                peak = max(peak, rss)
                mem = available_memory()
                disk = shutil.disk_usage("/tmp").free
                if stop_reason is None:
                    if STOP_REQUESTED or (ROOT / "STOP").exists() or now >= cutoff(plan) - 30:
                        stop_reason = "termination signal or absolute cutoff"
                    elif rss > limit:
                        stop_reason = "route RSS guard"
                    elif mem < headroom:
                        stop_reason = "host memory headroom guard"
                    elif disk < scratch_reserve:
                        stop_reason = "30 GiB scratch reserve"
                    if stop_reason:
                        kill_at = min(now + 8, cutoff(plan))
                        signal_group(proc.pid, signal.SIGTERM)
                elif now >= kill_at:
                    signal_group(proc.pid, signal.SIGKILL)
                if now - last_write >= 30 or stop_reason:
                    cores = None if last_sample is None else max(0, cpu - last_sample[1]) / (now - last_sample[0])
                    status.update(observed_unix=now, actual_CPU_seconds=cpu, actual_CPU_cores_interval=cores, current_RSS_bytes=rss,
                                  sampled_peak_RSS_bytes=peak, MemAvailable_bytes=mem,
                                  scratch_free_bytes=disk, stop_reason=stop_reason)
                    write(path, status)
                    last_write = now
                    last_sample = (now, cpu)
                time.sleep(1 if stop_reason else min(5, max(1, cutoff(plan) - now)))
            status.update(state="complete" if proc.returncode == 0 and not stop_reason else "failed",
                          returncode=proc.returncode, stop_reason=stop_reason,
                          actual_CPU_seconds=usage.ru_utime + usage.ru_stime,
                          sampled_peak_RSS_bytes=peak, child_peak_RSS_bytes=usage.ru_maxrss * 1024)
    except BaseException as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        if "proc" in locals() and proc.returncode is None:
            signal_group(proc.pid, signal.SIGKILL)
            _, wait_status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(wait_status)
            status.update(returncode=proc.returncode, actual_CPU_seconds=usage.ru_utime + usage.ru_stime,
                          child_peak_RSS_bytes=usage.ru_maxrss * 1024)
        status.update(finished_unix=time.time(), wall_seconds=time.time() - start)
        write(path, status)
    if status["state"] != "complete":
        raise RuntimeError(f"{label} {phase} failed. See {path}")


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def worker(plan, manifest, label):
    route = plan["routes"][label]
    job = os.environ["SLURM_JOB_ID"]
    if socket.gethostname().split(".")[0] != route["node"] or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Slurm host or single-core binding differs from the plan")
    if os.environ.get("LAVA_RUN_ROOT") != str(ROOT):
        raise RuntimeError("Unexpected run root")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    sys.path.insert(0, str(ROOT))
    from replay import completed_receipt, first_passed

    attempt = OPS / "attempts" / f"{label}-{job}-{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "WORKER.json", {"label": label, "job_id": job, "node": route["node"],
                                  "started_utc": utc_now(), "allocated_CPUs": 1})
    if label == "sp25" and not first_passed(manifest):
        run_phase(plan, label, "first", attempt)
    if not first_passed(manifest):
        raise RuntimeError("Both SP25 parity and its valid 300-epoch receipt are required")
    dispatch_errors = {}
    if label == "sp25":
        for child_label in plan["follow_on_order"]:
            try:
                submit_route(plan, child_label)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                dispatch_errors[child_label] = str(error)
        prefix = SCRATCH / "seeded_banks_v2" / (route["key"] + "__init161101") / "prefix500.json"
        write(attempt / "GATE_RELEASE.json", {
            "recorded_utc": utc_now(), "parity_and_first_fit_verified": True,
            "node": route["node"], "SP25_prefix500_present": prefix.is_file(), "dispatch_errors": dispatch_errors,
        })
    items = [item for item in manifest["jobs"].values() if item["job"]["required_data_key"] == route["key"]]
    if not all(completed_receipt(item) for item in items):
        run_phase(plan, label, "remaining", attempt)
    if not all(completed_receipt(item) for item in items):
        raise RuntimeError("Route ended without all its valid timing receipts")
    write(attempt / "ROUTE_COMPLETE.json", {"fits": len(items), "recorded_utc": utc_now()})
    if first_passed(manifest) and all(completed_receipt(item) for item in manifest["jobs"].values()):
        write(ROOT / "COMPLETE.json", {"fits": 19, "panels": 36, "epochs_each": 300,
                                      "manifest_sha256": plan["manifest_sha256"], "recorded_utc": utc_now()})
    if dispatch_errors:
        raise RuntimeError("One or more follow-on submissions require reconciliation")


def main():
    plan, manifest = load_plan()
    action = sys.argv[1:]
    if action == ["submit-first"]:
        if not (OPS / "DEPLOYMENT_VERIFIED.json").is_file():
            raise RuntimeError("Verified deployment receipt is required")
        print(json.dumps(submit_route(plan, "sp25")))
    elif len(action) == 2 and action[0] == "worker":
        worker(plan, manifest, action[1])
    else:
        raise SystemExit("Use submit-first or worker ROUTE")


if __name__ == "__main__":
    main()
