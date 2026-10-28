"""Budgeted policy iteration: bounded datagen, two fit epochs, paired strength."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from policy_ab import summary


class BudgetExpired(Exception):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(binary, *args):
    return ([sys.executable, str(binary)] if binary.suffix == ".py" else [str(binary)]) + list(map(str, args))


class Processes:
    def __init__(self, deadline):
        self.deadline = deadline
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.active = set()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired()
        if self.stopping.is_set():
            raise RuntimeError("another worker failed")

    @staticmethod
    def terminate(process):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        except ProcessLookupError:
            process.wait()

    def cancel(self):
        self.stopping.set()
        with self.lock:
            active = list(self.active)
        for process in active:
            self.terminate(process)

    def run(self, args, config, log, memory=0):
        self.check()
        if memory:
            if os.name != "posix":
                raise RuntimeError("fit memory enforcement requires POSIX; use --fit-memory-bytes 0 only under an external limit")
            args = [sys.executable, str(Path(__file__).resolve()), "--limit-child", str(memory), *args]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.with_suffix(".stdout.log").open("w") as out, log.with_suffix(".stderr.log").open("w") as err:
            process = subprocess.Popen(args, env=dict(os.environ, NANA_CONFIG=str(config)),
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       start_new_session=os.name == "posix")
            with self.lock:
                self.active.add(process)
            try:
                while process.poll() is None:
                    self.check()
                    try:
                        process.wait(timeout=min(.2, max(.001, self.deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError(f"{args[0]} exited {process.returncode}; see {log.with_suffix('.stderr.log')}")
            finally:
                self.terminate(process)
                with self.lock:
                    self.active.discard(process)
        return log.with_suffix(".stdout.log").read_text()


class Iterator:
    def __init__(self, options):
        self.o = options
        self.root = options.out.resolve()
        self.data = self.root / "data"
        self.lock = threading.Lock()
        inputs = dict(nana_sha256=sha(options.nana), trainer_sha256=sha(options.trainer),
                      config_sha256=sha(options.config),
                      initial_model_sha256=sha(options.initial_model) if options.initial_model else None)
        parameters = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(options).items()
                      if k not in ("out", "resume")}
        if not options.resume:
            self.root.mkdir(parents=True, exist_ok=False)
        self.run_lock = (self.root / ".run-lock").open("a+b")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self.run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                if not self.run_lock.tell():
                    self.run_lock.write(b"0")
                    self.run_lock.flush()
                self.run_lock.seek(0)
                msvcrt.locking(self.run_lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.run_lock.close()
            raise RuntimeError("an iterator already owns this output directory")
        if options.resume:
            manifest = json.loads((self.root / "manifest.json").read_text())
            if manifest["inputs"] != inputs or manifest["parameters"] != parameters:
                raise RuntimeError("resume inputs/options differ from the original run")
            self.token = manifest["owner"]
            self.state = json.loads((self.root / "state.json").read_text())
        else:
            self.token = uuid.uuid4().hex
            manifest = dict(inputs=inputs, parameters=parameters, owner=self.token,
                            start_unix=time.time(), epochs_per_iteration=2,
                            data_cap_bytes=int(options.data_gb * 1_000_000_000))
            atomic_json(self.root / "manifest.json", manifest)
            base = json.loads(options.config.read_text())
            base.update(policy="rank", policy_model="")
            atomic_json(self.root / "common.json", base)
            initial = None
            if options.initial_model:
                initial = self.root / "initial-policy.json"
                shutil.copyfile(options.initial_model, initial)
            self.state = dict(model=str(initial) if initial else None, completed_iterations=0)
            atomic_json(self.root / "state.json", self.state)
        if (self.root / ".policy-iterator-owner").exists():
            if (self.root / ".policy-iterator-owner").read_text() != self.token:
                raise RuntimeError("iterator ownership marker mismatch")
        elif options.resume:
            raise RuntimeError("missing iterator ownership marker")
        else:
            (self.root / ".policy-iterator-owner").write_text(self.token)
        self.base = json.loads((self.root / "common.json").read_text())
        # A completed per-iteration commit wins if interruption preceded state.json.
        commits = [json.loads(p.read_text()) for p in sorted(self.root.glob("iteration-*/iteration.json"))]
        if commits:
            self.state = dict(model=commits[-1]["model"], completed_iterations=len(commits))
            atomic_json(self.root / "state.json", self.state)
            if options.resume:
                with (self.root / "history.jsonl").open("w") as history:
                    for row in commits:
                        history.write(json.dumps(row) + "\n")
        remaining = manifest["start_unix"] + options.budget_hours * 3600 - time.time()
        self.processes = Processes(time.monotonic() + max(0, remaining))
        self.cap = manifest["data_cap_bytes"]
        self.block = os.statvfs(self.root).f_frsize if hasattr(os, "statvfs") else 4096

    def status(self, phase, **extra):
        atomic_json(self.root / "status.json", dict(phase=phase, model=self.state["model"],
                    completed_iterations=self.state["completed_iterations"],
                    remaining_seconds=max(0, self.processes.deadline - time.monotonic()), **extra))

    def reset_data(self):
        # Only the explicitly owned raw-data buffer may be recursively removed.
        expected = self.root / "data"
        if (self.data.is_symlink() or self.data.resolve() != expected or
                (self.root / ".policy-iterator-owner").read_text() != self.token):
            raise RuntimeError("refusing to reset an unowned data buffer")
        if self.data.exists():
            marker = self.data / ".policy-iterator-owner"
            if not marker.is_file() or marker.read_text() != self.token:
                raise RuntimeError("data buffer ownership marker missing or changed")
            shutil.rmtree(self.data)
        self.data.mkdir()
        (self.data / ".policy-iterator-owner").write_text(self.token)

    def config(self, directory, name, model):
        path = directory / f"{name}.json"
        atomic_json(path, dict(self.base, policy="softmax" if model else "rank", policy_model=model or ""))
        return path

    def collect(self, directory, config, seed0):
        used = reserved = next_game = complete = 0
        active = 0
        capped = threading.Event()
        started = time.monotonic()
        logs = directory / "datagen"
        with (directory / "datagen.jsonl").open("w", buffering=1) as game_log:
            def worker(cpu):
                nonlocal used, reserved, next_game, complete, active
                try:
                    while not capped.is_set():
                        self.processes.check()
                        with self.lock:
                            if next_game >= self.o.datagen_games:
                                return
                            available = (self.cap - self.block - used - reserved) // self.block * self.block
                            if available < 65536:
                                if not active:
                                    capped.set()
                                    return
                                limit = 0
                            else:
                                limit = min(1 << 30, available)
                                seed = seed0 + next_game
                                next_game += 1
                                reserved += limit
                                active += 1
                        if not limit:
                            time.sleep(.05)
                            continue
                        output = self.processes.run(command(self.o.nana, "--threads", 1, "--cpu", cpu,
                            "--games", 1, "--seed", seed, "--search-seed", 1, "--sims", self.o.sims,
                            "--max-pieces", self.o.datagen_pieces, "--tree-dump-dir", self.data,
                            "--tree-dump-max-bytes", limit), config, logs / f"game-{seed}")
                        row = json.loads(output)
                        path = self.data / f"tree-{seed}.jsonl"
                        size = path.stat().st_size
                        if size > limit:
                            raise RuntimeError("emulator exceeded its reserved dump budget")
                        with self.lock:
                            reserved -= limit
                            active -= 1
                            if row.get("tree_dump_capped", False):
                                # Do not train on a prefix selected by unordered-map iteration.
                                path.unlink()
                                capped.set()
                            else:
                                used += (size + self.block - 1) // self.block * self.block
                                complete += 1
                                game_log.write(json.dumps(row) + "\n")
                            self.status("datagen", iteration=directory.name, games=complete,
                                        data_bytes=used, reserved_bytes=reserved, data_cap_bytes=self.cap)
                except BaseException:
                    self.processes.stopping.set()
                    raise

            with ThreadPoolExecutor(max_workers=len(self.o.cpus)) as pool:
                jobs = [pool.submit(worker, cpu) for cpu in self.o.cpus]
                for job in jobs:
                    job.result()
        seeds = [int(p.stem[5:]) for p in self.data.glob("tree-*.jsonl")]
        if not any(s % 10 == 0 for s in seeds) or not any(s % 10 != 0 for s in seeds):
            raise RuntimeError("data buffer lacks complete training/validation games")
        return dict(games=complete, allocated_dump_bytes=used, cap_bytes=self.cap,
                    capped=capped.is_set(), seconds=time.monotonic() - started)

    def strength(self, directory, incumbent, candidate, seed0):
        configs = dict(rank=self.config(directory, "rank", None),
                       incumbent=self.config(directory, "incumbent", incumbent),
                       candidate=self.config(directory, "candidate", candidate))
        pairs = []
        with (directory / "strength.jsonl").open("w", buffering=1) as log:
            def worker(cpu, index):
                try:
                    for offset in range(index, self.o.strength_games, len(self.o.cpus)):
                        self.processes.check()
                        seed = seed0 + offset
                        row = dict(seed=seed)
                        order = ["rank", "incumbent", "candidate"]
                        rotation = (offset + offset // len(self.o.cpus)) % 3
                        order = order[rotation:] + order[:rotation]
                        for arm in order:
                            args = command(self.o.nana, "--threads", 1, "--cpu", cpu, "--games", 1,
                                           "--seed", seed, "--search-seed", 1, "--max-pieces", self.o.strength_pieces)
                            args += ["--sims", str(self.o.sims)] if not self.o.strength_ms else ["--ms", str(self.o.strength_ms)]
                            row[arm] = json.loads(self.processes.run(args, configs[arm],
                                directory / "strength" / f"{seed}-{arm}"))
                        with self.lock:
                            pairs.append(row)
                            log.write(json.dumps(row) + "\n")
                            self.status("strength", iteration=directory.name, pairs=len(pairs))
                except BaseException:
                    self.processes.stopping.set()
                    raise

            with ThreadPoolExecutor(max_workers=len(self.o.cpus)) as pool:
                jobs = [pool.submit(worker, cpu, i) for i, cpu in enumerate(self.o.cpus)]
                for job in jobs:
                    job.result()
        comparisons = {}
        for arm in ("rank", "incumbent"):
            comparison = summary([dict(rank=row[arm], softmax=row["candidate"]) for row in pairs], self.o.strength_pieces)
            comparison["baseline"] = comparison.pop("rank")
            comparison["candidate"] = comparison.pop("softmax")
            comparisons[arm] = comparison
        return comparisons

    def run(self):
        try:
            while not self.o.max_iterations or self.state["completed_iterations"] < self.o.max_iterations:
                self.processes.check()
                existing = [int(p.name.split("-")[1]) for p in self.root.glob("iteration-*") if p.is_dir()]
                iteration = max(existing, default=-1) + 1
                directory = self.root / f"iteration-{iteration:05d}"
                directory.mkdir()
                self.reset_data()
                started = time.monotonic()
                seed = self.o.seed + iteration * 10_000_000
                incumbent = self.state["model"]
                config = self.config(directory, "teacher", incumbent)
                loaded = json.loads(self.processes.run(command(self.o.nana, "--print-config"),
                                                       config, directory / "loaded"))
                atomic_json(directory / "teacher-loaded.json", loaded)
                if loaded.get("eval", "nana") != "nana" or any(loaded.get("belief_weights", [])):
                    raise RuntimeError("iterator requires nana eval with zero belief weights")
                self.status("datagen", iteration=directory.name)
                data = self.collect(directory, config, seed)
                self.status("fit", iteration=directory.name, data=data, epochs=2)
                fit = directory / "fit"
                args = command(self.o.trainer, self.data, fit, 2)
                if incumbent:
                    args.append(incumbent)
                self.processes.run(args, config, directory / "fit", self.o.fit_memory_bytes)
                metrics = json.loads((fit / "metrics.json").read_text())
                if [e["epoch"] for e in metrics["epochs"]] != [0, 1, 2]:
                    raise RuntimeError("fit did not complete exactly two epochs")
                candidate = str(fit / "policy.json")
                self.status("strength", iteration=directory.name, data=data)
                strength = self.strength(directory, incumbent, candidate, seed + 5_000_000)
                row = dict(iteration=iteration, model=candidate, model_sha256=sha(candidate),
                           incumbent=incumbent, data=data, fit=metrics, strength=strength,
                           seconds=time.monotonic() - started)
                atomic_json(directory / "iteration.json", row)
                with (self.root / "history.jsonl").open("a", buffering=1) as history:
                    history.write(json.dumps(row) + "\n")
                self.state = dict(model=candidate, completed_iterations=self.state["completed_iterations"] + 1)
                atomic_json(self.root / "state.json", self.state)
                atomic_json(self.root / "latest.json", dict(model=candidate, sha256=sha(candidate)))
                print(json.dumps(dict(iteration=iteration, model=candidate, strength=strength)), flush=True)
            self.status("complete")
        except BudgetExpired:
            self.processes.cancel()
            self.status("budget_expired")
        except BaseException as error:
            self.processes.cancel()
            self.status("failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nana", "trainer", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--budget-hours", type=float, required=True)
    parser.add_argument("--cpus", default="0,2,4,6,8,10,12,14")
    parser.add_argument("--data-gb", type=float, default=90)
    parser.add_argument("--datagen-games", type=int, default=1000)
    parser.add_argument("--datagen-pieces", type=int, default=50)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--strength-games", type=int, default=100)
    parser.add_argument("--strength-pieces", type=int, default=100)
    parser.add_argument("--strength-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10_000_000)
    parser.add_argument("--fit-memory-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    options.cpus = list(map(int, options.cpus.split(",")))
    if (not math.isfinite(options.budget_hours) or options.budget_hours <= 0 or
            not math.isfinite(options.data_gb) or options.data_gb < .0001 or
            not options.cpus or min(options.cpus) < 0 or len(set(options.cpus)) != len(options.cpus) or
            not 2 <= options.datagen_games < 5_000_000 or not 1 <= options.strength_games < 5_000_000 or
            min(options.datagen_pieces, options.sims, options.strength_pieces) < 1 or
            min(options.strength_ms, options.max_iterations, options.fit_memory_bytes, options.seed) < 0):
        parser.error("invalid budget, buffer size, CPUs, game count or search settings")
    for name in ("nana", "trainer", "config", "initial_model"):
        value = getattr(options, name)
        if value:
            setattr(options, name, value.resolve(strict=True))
    iterator = Iterator(options)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    iterator.run()


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--limit-child":
        import resource
        cap = int(sys.argv[2])
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
    main()
