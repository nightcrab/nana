#!/usr/bin/env python3
"""Run the C++ statistical diagnostics (nana_statcheck)."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/statcheck'))
    parser.add_argument('--trials', type=int, default=100000)
    args = parser.parse_args()
    if args.trials < 10000:
        parser.error('--trials must be at least 10000')
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = (20261024, 71393, 991827)
    reports = []
    failed = False
    for seed in seeds:
        run = subprocess.run([str(args.binary.resolve()), str(args.trials), str(seed)],
                             capture_output=True, text=True)
        (args.out / f'{seed}.json').write_text(run.stdout, encoding='utf-8')
        (args.out / f'{seed}.err').write_text(run.stderr, encoding='utf-8')
        try:
            report = json.loads(run.stdout)
        except json.JSONDecodeError:
            raise SystemExit(f'statcheck exited {run.returncode}; see {args.out / f"{seed}.err"}')
        reports.append(report)
        checks = [r for r in report['results'] if r['status'] != 'diagnostic']
        failed |= run.returncode != 0 or report['failures'] != 0
        print(f"seed {seed}: {len(checks)} checks, {report['failures']} failures", flush=True)
    comparisons = sum((9 if '/one-turn/' in r['name'] else 81)
                      for p in reports for r in p['results'] if 'samples' in r)
    if comparisons > 10000:
        raise SystemExit('comparison budget exceeded; update the statistical design')
    summary = {'failed': failed, 'seeds': list(seeds), 'trials': args.trials,
               'comparison_cells_upper_bound': comparisons, 'family_alpha': 0.001,
               'reports': reports}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for r in reports[0]['results']:
        if r['status'] == 'diagnostic':
            print(r['name'] + ': ' + json.dumps({k: v for k, v in r.items()
                                               if k not in ('name', 'status', 'interpretation')}))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
