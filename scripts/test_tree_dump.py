"""Integration checks: python scripts/test_tree_dump.py path/to/nana_emulate."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="nana-tree-") as temporary:
        root = Path(temporary)

        def run(*arguments, ok=True):
            result = subprocess.run(
                [executable, "--threads", "1", "--sims", "64", "--search-seed", "1",
                 "--max-pieces", "4", *map(str, arguments)],
                capture_output=True, text=True, timeout=60,
            )
            assert (result.returncode == 0) == ok, result.stderr
            return [json.loads(line) for line in result.stdout.splitlines()]

        def read(directory, seed):
            with (directory / f"tree-{seed}.jsonl").open() as stream:
                header = json.loads(next(stream))
                nodes = [json.loads(line) for line in stream]
            assert header["version"] == 1 and header["seed"] == seed
            assert header["min_node_N"] == 10
            assert nodes and len({node["id"] for node in nodes}) == len(nodes)
            for node in nodes:
                assert len(node["board"]) == 10 and len(node["queue"]) == 6
                assert node["N"] >= 10
                if node["actions"]:
                    assert math.isclose(sum(a["prior"] for a in node["actions"]), 1, abs_tol=1e-5)
                for action in node["actions"]:
                    assert action["N"] >= 0
                    assert all(math.isfinite(action[k]) for k in ("N", "R", "prior", "eval"))
            return {node["id"]: node for node in nodes}

        collected = run("--games", 2, "--seed", 100, "--tree-dump-dir", root / "batch")
        control = run("--games", 2, "--seed", 100)
        for before, after in zip(control, collected, strict=True):
            for key in ("pieces", "died", "attack", "lines", "sims_per_move", "opponent_attack"):
                assert before[key] == after[key], (key, before, after)
        read(root / "batch", 100)
        batch_second = read(root / "batch", 101)
        run("--games", 1, "--seed", 101, "--tree-dump-dir", root / "single")
        assert batch_second == read(root / "single", 101), "tree leaked across games"

        # Existing data must survive a repeated seed, and write errors must fail the run.
        path = root / "batch" / "tree-100.jsonl"
        original = path.read_bytes()
        run("--games", 1, "--seed", 100, "--tree-dump-dir", root / "batch", ok=False)
        assert path.read_bytes() == original
        run("--games", 1, "--tree-dump-dir", path / "invalid", ok=False)

        run("--games", 1, "--seed", 102, "--threads", 2,
            "--tree-dump-dir", root / "parallel")
        read(root / "parallel", 102)
    print("tree dump checks passed: schema, gameplay, reset, write failures, two threads")


if __name__ == "__main__":
    main()
