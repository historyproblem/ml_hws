"""Schedule independent suite jobs in isolated processes, preserving old runs.

This lives outside spend_baselines: existing model/source fingerprints remain
unchanged. Its own SHA256 and worker count are recorded as execution metadata.
"""

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import platform
import queue
import signal
import time
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from pathlib import Path

from spend_baselines.suite import (
    combine,
    execute_job,
    existing_result,
    identity,
    job_key,
    make_plan,
    read_config,
    write_scores,
)
from spend_baselines.suite_data import CachedData, atomic_json, digest, prepare_cache


def worker_loop(number, config, args, tasks, messages):
    # Only the coordinator handles Ctrl+C and terminates its own children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cache = CachedData(args.out_dir / "cache")
    logs = args.out_dir / "parallel_logs"
    logs.mkdir(exist_ok=True)
    while True:
        job = tasks.get()
        if job is None:
            return
        key = job_key(job)
        started = time.monotonic()
        log = logs / f"{key}.log"
        try:
            # Execution logs stay outside scientific artifacts/checksums.
            with (
                log.open("w", buffering=1) as stream,
                redirect_stdout(stream),
                redirect_stderr(stream),
            ):
                print(
                    f"worker={number} pid={os.getpid()} job={json.dumps(job, sort_keys=True)}",
                    flush=True,
                )
                result = execute_job(job, config, cache, args)
            messages.put(
                {
                    "worker": number,
                    "job": job,
                    "result": result,
                    "seconds": time.monotonic() - started,
                    "pid": os.getpid(),
                }
            )
        except Exception as error:  # noqa: BLE001 - relay job failures across the process boundary
            messages.put(
                {
                    "worker": number,
                    "job": job,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                }
            )


def runtime_definition(config, data_dir, device, threads):
    # Exact schema of the serial runner: no migration or ignored differences.
    return {
        "definition": identity(config, data_dir),
        "device": device,
        "threads": threads,
        "python": platform.python_version(),
        "libraries": {
            n: version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "torch")
        },
    }


def execute_parallel(config, cache, args, plan, selected, results):
    pending = [j for j in selected if job_key(j) not in results]
    failures = []
    events = []
    started = time.monotonic()
    context = mp.get_context("spawn")
    messages = context.Queue()
    processes = []
    tasks = []
    active = {}
    iterator = iter(pending)
    worker_count = min(args.workers, len(pending))
    interrupted = False
    failure = None

    def checkpoint():
        ordered = [results[job_key(j)] for j in plan if job_key(j) in results]
        write_scores(ordered, args.out_dir)
        atomic_json(
            args.out_dir / "status.json",
            {
                "planned": len(plan),
                "completed": len(ordered),
                "remaining": len(plan) - len(ordered),
                "failed_in_this_invocation": failures,
                "workers": worker_count,
                "interrupted": interrupted,
                "active": [job_key(j) for j in active.values()],
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    print(
        f"Verified {len(results)}/{len(plan)} completed jobs; scheduling {len(pending)} jobs on {worker_count} workers",
        flush=True,
    )
    try:
        for number in range(worker_count):
            task = context.Queue(maxsize=1)
            process = context.Process(
                target=worker_loop, args=(number, config, args, task, messages)
            )
            process.start()
            tasks.append(task)
            processes.append(process)

        def assign(number):
            job = next(iterator, None)
            if job is None:
                tasks[number].put(None)
                active.pop(number, None)
                return
            active[number] = job
            tasks[number].put(job)
            print(
                f"START worker={number} {job_key(job)} {json.dumps(job, sort_keys=True)}",
                flush=True,
            )

        for number in range(worker_count):
            assign(number)
        while active:
            try:
                message = messages.get(timeout=1)
            except queue.Empty:
                dead = [
                    number
                    for number in active
                    if processes[number].exitcode is not None
                ]
                if dead:
                    raise RuntimeError(
                        f"Workers exited unexpectedly: {dead}; unfinished jobs remain resumable"
                    )
                continue
            number = message["worker"]
            job = active.pop(number)
            if job != message["job"]:
                raise RuntimeError("Worker returned a different job")
            key = job_key(job)
            if "result" in message:
                results[key] = message["result"]
                score = message["result"].get(
                    "Brier" if job.get("task") == "probability" else "RMSLE"
                )
                print(
                    f"DONE {len(results)}/{len(plan)} worker={number} {key} metric={score} seconds={message['seconds']:.1f}",
                    flush=True,
                )
            else:
                failures.append(key)
                directory = args.out_dir / "jobs" / key
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "error.json", message)
                print(f"FAILED worker={number} {key}: {message['error']}", flush=True)
                if not config.get("continue_on_error", False):
                    raise RuntimeError(message["error"])
            events.append({k: message[k] for k in ("worker", "pid", "job")})
            assign(number)
            checkpoint()
        for process in processes:
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(f"Worker did not exit normally: {process.exitcode}")
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping workers; completed job artifacts are preserved", flush=True)
    except BaseException as error:  # noqa: BLE001 - persist status, stop children, then re-raise
        failure = error
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        # A worker may have committed a result just before interruption.
        for job in plan:
            result = existing_result(args.out_dir / "jobs" / job_key(job), job)
            if result is not None:
                results[job_key(job)] = result
        active.clear()
        checkpoint()
        atomic_json(
            args.out_dir / "parallel_execution.json",
            {
                "launcher_sha256": digest(Path(__file__)),
                "workers": args.workers,
                "started_pending": len(pending),
                "completed_total": len(results),
                "seconds": time.monotonic() - started,
                "interrupted": interrupted,
                "error": None if failure is None else str(failure),
                "events": events,
            },
        )
        for task in tasks:
            task.close()
            task.cancel_join_thread()
        messages.close()
        messages.cancel_join_thread()
    if failure is not None:
        raise failure
    if interrupted:
        return 130
    ordered = [results[job_key(j)] for j in plan if job_key(j) in results]
    count = combine(ordered, config, cache, args.out_dir)
    print(
        f"Completed {len(results)}/{len(plan)} jobs; {count} derived evaluations; failures={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v100_suite.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/v100-focused"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--kinds", nargs="+", choices=["tabular", "neural", "bagging", "clv"]
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.threads < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("workers, threads and limit must be positive")
    args.resume = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        free, total = torch.cuda.mem_get_info()
        print(
            f"GPU {torch.cuda.get_device_name(0)}; free={free / 2**30:.1f}/{total / 2**30:.1f} GiB",
            flush=True,
        )
    config = read_config(args.config, args.smoke)
    plan = make_plan(config)
    selected = [j for j in plan if not args.kinds or j["kind"] in args.kinds]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        parser.error("Filters selected no jobs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "A suite process is already running in this output directory; stop it before resuming in parallel"
            ) from error
        runtime = runtime_definition(config, args.data_dir, args.device, args.threads)
        marker = args.out_dir / "run_manifest.json"
        if marker.exists() and json.loads(marker.read_text()) != runtime:
            raise ValueError(
                "Run code/data/config/runtime changed; refusing to reuse its artifacts"
            )
        cache = prepare_cache(
            args.data_dir, args.out_dir / "cache", config, runtime["definition"]
        )
        if any(j.get("representation") == "F4" for j in selected):
            from spend_baselines.suite_clv import prepare_clv

            prepare_clv(cache, args.data_dir, config["clv"], runtime["definition"])
        if not marker.exists():
            atomic_json(marker, runtime)
        atomic_json(args.out_dir / "plan.json", plan)
        results = {}
        for job in plan:
            result = existing_result(args.out_dir / "jobs" / job_key(job), job)
            if result is not None:
                results[job_key(job)] = result
        print(
            json.dumps(
                {
                    "planned": len(plan),
                    "by_kind": Counter(j["kind"] for j in plan),
                    "workers": args.workers,
                    "model_math": "unchanged",
                }
            ),
            flush=True,
        )
        return execute_parallel(config, cache, args, plan, selected, results)


if __name__ == "__main__":
    raise SystemExit(main())
