"""Check runtime priors against an independent softmax of the trained artifact."""
import copy
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main():
    executable, artifact = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    model = json.loads(artifact.read_text())
    with tempfile.TemporaryDirectory(prefix="nana-softmax-") as temporary:
        root = Path(temporary)
        config = root / "config.json"
        model_file = root / "policy.json"

        def run(policy, data=model, ok=True):
            config.write_text(json.dumps({"policy": policy, "policy_model": "policy.json" if policy == "softmax" else ""}))
            model_file.write_text(json.dumps(data))
            process = subprocess.run([str(executable)], env=dict(os.environ, NANA_CONFIG=str(config)),
                                     capture_output=True, text=True, timeout=60)
            assert (process.returncode == 0) == ok, process.stderr
            return [json.loads(line) for line in process.stdout.splitlines()] if ok else None

        baseline = run("rank")
        candidate = run("softmax")
        assert [len(n) for n in baseline] == [len(n) for n in candidate]
        assert len(candidate) == 3 and all(candidate)
        for old, new in zip(baseline, candidate, strict=True):
            logits = []
            for a, b in zip(old, new, strict=True):
                assert a["eval"] == b["eval"] and a["move_reward"] == b["move_reward"]
                assert a["features"] == b["features"]
                scaled = [struct.unpack("f", struct.pack("f", x / scale))[0]
                          for x, scale in zip(b["features"], model["feature_scales"], strict=True)]
                logits.append(sum(x * w for x, w in zip(scaled, model["scaled_weights"], strict=True)))
            weights = [math.exp(x - max(logits)) for x in logits]
            for action, w in zip(new, weights, strict=True):
                assert math.isclose(action["prior"], w / sum(weights), rel_tol=2e-6, abs_tol=1e-8)
            assert math.isclose(sum(a["prior"] for a in new), 1, abs_tol=1e-6)
        zero = copy.deepcopy(model)
        zero["scaled_weights"] = [0] * len(model["scaled_weights"])
        for node in run("softmax", zero):
            assert all(math.isclose(a["prior"], 1 / len(node), abs_tol=1e-7) for a in node)
        shifted = copy.deepcopy(model)
        for i in range(26, 31):
            shifted["scaled_weights"][i] += 10000
        for expected, actual in zip(candidate, run("softmax", shifted), strict=True):
            assert all(math.isclose(a["prior"], b["prior"], abs_tol=1e-7)
                       for a, b in zip(expected, actual, strict=True))
        for key, value in (("scaled_weights", [0]), ("feature_names", ["wrong"]), ("max_well_depth", -1)):
            invalid = copy.deepcopy(model)
            invalid[key] = value
            run("softmax", invalid, ok=False)
    print("softmax policy checks passed: trained probabilities, unchanged values/rewards, uniform logits, stability, validation")


if __name__ == "__main__":
    main()
