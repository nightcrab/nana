"""Check leaf ordering and normalized rank-power priors against engine output."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    check, emulator = (str(Path(p).resolve()) for p in sys.argv[1:3])
    clean_env = {k: v for k, v in os.environ.items() if k != "NANA_CONFIG"}
    result = subprocess.run([emulator, "--print-config"], env=clean_env,
                            text=True, capture_output=True, check=True, timeout=30)
    base = json.loads(result.stdout)
    assert not any(key.startswith("policy") for key in base)
    with tempfile.TemporaryDirectory(prefix="nana-rank-") as temporary:
        config_path = Path(temporary) / "config.json"

        def run(config, valid=True):
            config_path.write_text(json.dumps(config))
            output = subprocess.run([check], env=dict(clean_env, NANA_CONFIG=str(config_path)),
                                    text=True, capture_output=True, timeout=30)
            assert (output.returncode == 0) == valid, output.stderr
            return [json.loads(line) for line in output.stdout.splitlines()] if valid else None

        def verify(nodes, power):
            for actions in nodes:
                assert actions
                ordered = sorted(actions, key=lambda a: a["eval"], reverse=True)
                weights = [rank ** -power for rank in range(1, len(actions) + 1)]
                total = sum(weights)
                # Equal leaf values may take any order within their rank group.
                offset = 0
                while offset < len(ordered):
                    end = offset + 1
                    while end < len(ordered) and ordered[end]["eval"] == ordered[offset]["eval"]:
                        end += 1
                    actual = sorted(a["prior"] for a in ordered[offset:end])
                    expected = sorted(w / total for w in weights[offset:end])
                    assert all(math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-8)
                               for a, b in zip(actual, expected, strict=True))
                    offset = end
                assert math.isclose(sum(a["prior"] for a in actions), 1, abs_tol=2e-6)

        baseline = run(base)
        verify(baseline, base["prior_exponent"])
        for power in (0, 0.5, 2):
            nodes = run(dict(base, prior_exponent=power))
            verify(nodes, power)
            for old, new in zip(baseline, nodes, strict=True):
                assert [(a["eval"], a["move_reward"]) for a in old] == [
                    (a["eval"], a["move_reward"]) for a in new]
        changed = run(dict(base, height=100000))
        verify(changed, base["prior_exponent"])
        assert any(a["prior"] != b["prior"] for old, new in zip(baseline, changed, strict=True)
                   for a, b in zip(old, new, strict=True)), "leaf weights must change ordering"
        for key, value in (("policy", "softmax"), ("policy", "learned-rank"),
                           ("policy_model", "removed.json"), ("policy_weights", [0] * 43),
                           ("policy_leaf_eval", True), ("policy_leaf_weights", [0] * 48),
                           ("policy_max_well_depth", 17)):
            run(dict(base, **{key: value}), valid=False)
    print("Rank policy checks passed: leaf ordering, rank power, normalization, unchanged rewards, removed settings")


if __name__ == "__main__":
    main()
