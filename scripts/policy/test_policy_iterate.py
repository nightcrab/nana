"""Fast orchestration tests with subprocess fixtures; no remote host required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from policy_iterate import Iterator


FAKE = r'''
import json, os, sys, time
from pathlib import Path
a=sys.argv[1:]
if a and not a[0].startswith('--'):
    data, output, epochs = Path(a[0]), Path(a[1]), int(a[2])
    assert epochs == 2
    if 'NANA_FAKE_REQUIRE_LIMIT' in os.environ:
        import resource
        assert resource.getrlimit(resource.RLIMIT_AS)[0] == int(os.environ['NANA_FAKE_REQUIRE_LIMIT'])
    initial=json.loads(Path(a[3]).read_text()) if len(a)>3 else {'generation':0}
    time.sleep(float(os.environ.get('NANA_FAKE_FIT_SLEEP','0')))
    output.mkdir()
    generation=initial['generation']+1
    (output/'policy.json').write_text(json.dumps({'generation':generation}))
    seeds=[int(p.stem[5:]) for p in data.glob('tree-*.jsonl')]
    (output/'metrics.json').write_text(json.dumps({'epochs':[{'epoch':i} for i in range(3)],
        'initial_generation':initial['generation'],'seeds':seeds,'best_epoch':2}))
    sys.exit(0)
config=json.loads(Path(os.environ['NANA_CONFIG']).read_text())
if '--print-config' in a:
    print(json.dumps(config)); sys.exit(0)
def value(key): return a[a.index(key)+1]
generation=json.loads(Path(config['policy_model']).read_text())['generation'] if config.get('policy_model') else 0
seed=int(value('--seed'))
capped=False
if '--tree-dump-dir' in a:
    path=Path(value('--tree-dump-dir'))/f'tree-{seed}.jsonl'
    limit=int(value('--tree-dump-max-bytes'))
    content=json.dumps({'kind':'header','padding':'x'*59900})+'\n'
    capped=len(content)>limit
    path.write_bytes(content.encode()[:limit])
print(json.dumps({'seed':seed,'pieces':int(value('--max-pieces')),'died':False,
    'attack':generation+1,'sims_per_move':64,'ms_per_move':1,'tree_dump_capped':capped}))
'''


def main():
    script = Path(__file__).with_name("policy_iterate.py")
    with tempfile.TemporaryDirectory(prefix="nana-iterator-") as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(FAKE)
        config = root / "config.json"
        config.write_text("{}")
        initial = root / "initial.json"
        initial.write_text('{"generation":0}')

        def args(out, budget="1", cap=".001", games="4"):
            return [sys.executable, str(script), "--nana", str(fake), "--trainer", str(fake),
                    "--config", str(config), "--initial-model", str(initial), "--out", str(out),
                    "--budget-hours", budget, "--cpus", "0,1", "--data-gb", cap,
                    "--datagen-games", games, "--datagen-pieces", "4", "--sims", "64",
                    "--strength-games", "2", "--strength-pieces", "4", "--fit-memory-bytes", "0",
                    "--max-iterations", "2"]

        out = root / "run"
        subprocess.run(args(out), check=True, stdout=subprocess.DEVNULL, timeout=30)
        history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
        assert len(history) == 2
        assert [r["fit"]["initial_generation"] for r in history] == [0, 1]
        assert not set(history[0]["fit"]["seeds"]) & set(history[1]["fit"]["seeds"])
        final_seeds = {int(p.stem[5:]) for p in (out / "data").glob("tree-*.jsonl")}
        assert final_seeds == set(history[1]["fit"]["seeds"]), "old data was retained"
        assert json.loads((out / "status.json").read_text())["completed_iterations"] == 2
        # Recover a completed commit even if the global state update was interrupted.
        (out / "state.json").write_text(json.dumps(dict(model=None, completed_iterations=0)))
        subprocess.run([*args(out), "--resume"], check=True, stdout=subprocess.DEVNULL, timeout=10)
        assert len((out / "history.jsonl").read_text().splitlines()) == 2
        assert json.loads((out / "status.json").read_text())["completed_iterations"] == 2

        capped = root / "capped"
        subprocess.run(args(capped, cap=".0002", games="10"), check=True, stdout=subprocess.DEVNULL, timeout=30)
        rows = [json.loads(line) for line in (capped / "history.jsonl").read_text().splitlines()]
        assert all(r["data"]["capped"] and r["data"]["allocated_dump_bytes"] <= 200000 for r in rows)
        assert sum(p.stat().st_size for p in (capped / "data").glob("tree-*")) <= 200000

        expired = root / "expired"
        started = time.monotonic()
        active = subprocess.Popen(args(expired, budget=".0005"), env=dict(os.environ, NANA_FAKE_FIT_SLEEP="10"),
                                  stdout=subprocess.DEVNULL)
        while not (expired / "status.json").exists() and active.poll() is None:
            time.sleep(.02)
        duplicate = subprocess.run([*args(expired, budget=".0005"), "--resume"], capture_output=True, timeout=5)
        assert duplicate.returncode != 0 and b"already owns" in duplicate.stderr
        assert active.wait(timeout=10) == 0
        status = json.loads((expired / "status.json").read_text())
        assert status["phase"] == "budget_expired" and status["completed_iterations"] == 0
        assert not (expired / "latest.json").exists(), "partial fit was promoted"
        assert time.monotonic() - started < 6

        if os.name == "posix":
            limited = args(root / "limited")
            limited[limited.index("--fit-memory-bytes") + 1] = "2000000000"
            subprocess.run(limited, env=dict(os.environ, NANA_FAKE_REQUIRE_LIMIT="2000000000"),
                           check=True, stdout=subprocess.DEVNULL, timeout=30)

        # A data-directory symlink must never cause deletion outside the owned buffer.
        outside = root / "outside"
        outside.mkdir()
        sentinel = outside / "keep"
        sentinel.write_text("keep")
        link = root / "guarded" / "data"
        link.parent.mkdir()
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pass  # Windows may disallow symlink creation without developer mode.
        else:
            iterator = Iterator.__new__(Iterator)
            iterator.root, iterator.data, iterator.token = link.parent, link, "test"
            try:
                iterator.reset_data()
                raise AssertionError("unsafe buffer reset accepted")
            except RuntimeError:
                assert sentinel.read_text() == "keep"
    print("policy iterator checks passed: two cycles, warm start, overwrite, hard cap, deadline, resume, ownership")


if __name__ == "__main__":
    main()
