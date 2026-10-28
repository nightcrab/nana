# Trained softmax prior

Set `policy` to `softmax` and `policy_model` to a fitted artifact:

```json
{
  "policy": "softmax",
  "policy_model": "trained-policy-20261028.json"
}
```

Model paths are relative to the configuration file (or executable when there is
no configuration file). Default `policy: rank` preserves the rank-based prior.
The softmax option requires `eval: nana`. Its 43-feature basis and float feature
conversion are shared with the C++ trainer; model version, names, scales and
weight dimensions are checked on load. Softmax subtracts the maximum logit
before exponentiation. It does not apply the rank power law or add smoothing.

The model affects only edge priors. Static evaluations, leaf values, move rewards
and backup settings continue using the ordinary evaluator configuration. This
first implementation computes the policy's placement features separately, so
fixed-time testing matters as well as fixed-simulation testing.

`tuner/releases/trained-policy-20261028.json` is the epoch-2 checkpoint fitted
to 900 games, selected on 100 held-out games, from the original default teacher.
It is an opt-in policy model; the current default value release is a different,
subsequently tuned evaluator.

Build with `NANA_EVALCHECK=ON` and run:

```text
python scripts/test_softmax_policy.py path/to/nana_policycheck tuner/releases/trained-policy-20261028.json
```

Run a paired A/B with identical value settings and fresh seeds:

```text
python scripts/policy_ab.py --nana build/cpp/nana_emulate --model tuner/releases/trained-policy-20261028.json --config tuner/releases/unified-cc-20261028.json --out results/policy-ab --games 1000 --wall-games 200 --sims 1000 --ms 50 --horizon 100 --seed 1000000 --cpus 0,2,4,6,8,10,12,14
```

Each seed is played by both priors on the same CPU, alternating arm order.
Fixed-time seeds are disjoint from fixed-simulation seeds. Fitness is attack
divided by the fixed piece horizon, counting unplayed pieces after death as zero
attack. `pairs.jsonl` stores raw game results; `status.json` reports paired mean
differences, approximate 95% confidence intervals, survival and throughput.
The manifest records binary, model and common-configuration hashes. These are
emulated-opponent tests, not head-to-head matches or an automatic release gate.
