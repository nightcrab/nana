"""End-to-end reconstruction and fitting check, using two supplied executables."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    emulator, trainer = map(lambda p: str(Path(p).resolve()), sys.argv[1:3])
    with tempfile.TemporaryDirectory(prefix="nana-policy-fit-") as directory:
        root = Path(directory)
        subprocess.run(
            [emulator, "--threads", "1", "--games", "20", "--seed", "0",
             "--sims", "64", "--search-seed", "1", "--max-pieces", "4",
             "--tree-dump-dir", str(root / "dumps")],
            check=True, stdout=subprocess.DEVNULL, timeout=60,
        )
        subprocess.run([trainer, str(root / "dumps"), str(root / "fit"), "3"],
                       check=True, stdout=subprocess.DEVNULL, timeout=60)
        metrics = json.loads((root / "fit/metrics.json").read_text())
        model = json.loads((root / "fit/policy.json").read_text())
        assert metrics["train_games"] == 18 and metrics["validation_games"] == 2
        assert metrics["max_reconstruction_error"] < 2e-5
        assert metrics["checked_actions"] > 0
        initial, final = metrics["epochs"][0], metrics["epochs"][-1]
        assert final["validation"]["cross_entropy"] < initial["validation"]["cross_entropy"]
        assert len(model["raw_logit_weights"]) == 43
        assert all(math.isfinite(w) for w in model["raw_logit_weights"])
        assert any(abs(w) > 1e-6 for w in model["raw_logit_weights"])
        # Protect an existing fitted model from an accidental rerun.
        original = (root / "fit/policy.json").read_bytes()
        failure = subprocess.run([trainer, str(root / "dumps"), str(root / "fit"), "1"],
                                 capture_output=True, timeout=60)
        assert failure.returncode != 0
        assert (root / "fit/policy.json").read_bytes() == original
    print("policy fit checks passed: reconstruction, held-out learning, finite weights, overwrite protection")


if __name__ == "__main__":
    main()
