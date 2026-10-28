"""Set nana's CC search (nana_cctree) against Cold Clear (tree_dump) on the same position, and say
where and why their trees and answers part.

Both tools run one simulation at a time and draw their selections from the same splitmix64 stream,
and nana's search plays one fixed future past its preview, which tree_dump gets as a known queue.
With identical semantics the two trees would be identical, so the first step where they differ
names the cause: different candidate moves (movegen), different values (eval or backup), equal
values ranked in another order (ties), different weights from equal values (precision), or a
speculated piece (only with --speculate, where Cold Clear sees just the preview).

usage: python scripts/opponent/cc_tree_compare.py [--seed S] [--count K] [--pieces P] [--sims N]
           [--lockstep-seed L] [--speculate] [--no-time] [--canonical] [--clear-meter] [--nana EXE] [--cc EXE] [--out DIR] [--show N]

nana_cctree: cmake -DNANA_CCTREE=ON, target nana_cctree (NANA_CCTREE_LOAD_FACTOR, default 1).
tree_dump: cargo build --release -p cc-tbp --bin tree_dump --features tetrio_garbage,lockstep
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ".exe" if os.name == "nt" else ""
DEFAULT_NANA = ROOT / "out" / "build" / "cctree" / "cpp" / f"nana_cctree{EXE}"
DEFAULT_CC = ROOT.parent / "cold-clear" / "target" / "release" / f"tree_dump{EXE}"

# SRS cells in TBP coordinates (libtetris PieceState::cells)
NORTH = {
    "I": [(-1, 0), (0, 0), (1, 0), (2, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(-1, 0), (0, 0), (1, 0), (0, 1)],
    "L": [(-1, 0), (0, 0), (1, 0), (1, 1)],
    "J": [(-1, 0), (0, 0), (1, 0), (-1, 1)],
    "S": [(-1, 0), (0, 0), (0, 1), (1, 1)],
    "Z": [(-1, 1), (0, 1), (0, 0), (1, 0)],
}
ROTATE = {
    "north": lambda x, y: (x, y),
    "east": lambda x, y: (y, -x),
    "south": lambda x, y: (-x, -y),
    "west": lambda x, y: (-y, x),
}


def canon(name):
    """a placement by the cells it fills, whatever orientation or spin label names it"""
    t, rot, x, y, _ = name.split()
    x, y = int(x), int(y)
    return (t, tuple(sorted((x + dx, y + dy) for dx, dy in (ROTATE[rot](*c) for c in NORTH[t]))))


def spin(name):
    return name.split()[4]


def run(cmd, env=None):
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.exit(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr}")
    return json.loads(proc.stdout)


class Tree:
    """either tool's tree as nodes of {ck: edge}, edge = {name, value, reward, static, child}"""

    def __init__(self, dump):
        self.tool = dump["tool"]
        self.nodes = {}
        if self.tool == "nana":
            self.root = dump["root"]
            for n in dump["tree"]:
                edges = {}
                for c in n["children"]:
                    edges[canon(c["mv"])] = {
                        "name": c["mv"], "value": c["value"], "reward": c["reward"], "static": c["static"],
                        "child": c["child"], "game_over": c["game_over"],
                    }
                self.nodes[n["id"]] = {"value": n["value"], "edges": edges, "expanded": True}
        else:
            self.root = "0:0"
            raw = {n["id"]: n for n in dump["tree"]}
            for n in dump["tree"]:
                edges = {}
                for c in n["children"]:
                    child = raw.get(c["child"])
                    dead = child is None or child["death"]
                    edges[canon(c["mv"])] = {
                        "name": c["mv"], "value": None if dead else child["value"] + c["reward"],
                        "reward": c["reward"], "static": None,
                        "child": c["child"] if child is not None and child["expanded"] else None,
                        "piece": c["piece"],
                    }
                self.nodes[n["id"]] = {
                    "value": None if n["death"] else n["value"], "edges": edges, "expanded": n["expanded"],
                    "speculated": n["speculated"],
                }

    def best_edges(self, node, k):
        edges = [e for e in self.nodes[node]["edges"].values() if e["value"] is not None]
        return sorted(edges, key=lambda e: -e["value"])[:k]


def nana_thinks(dump):
    thinks = []
    for step in dump["trace"]:
        if step["node"] == dump["root"] or not thinks:
            thinks.append([])
        thinks[-1].append(step)
    return thinks


def fmt_ranked(ranked, n):
    return ", ".join(f"{r[0]} {r[1]} w{int(r[-1])}" for r in ranked[:n])


def compare_step(ns, cs):
    """differences between one nana select_CC call and Cold Clear's matching choice"""
    if "speculate" in cs:
        return ["speculation"], "Cold Clear drew a speculated piece here; nana's future is fixed"
    out = []
    detail = []
    nr = {canon(r[0]): r for r in ns["ranked"]}
    # Cold Clear lists deaths with weight 0; nana leaves them out
    cr = {canon(r[0]): r for r in cs["ranked"] if r[1] is not None}
    only_n = [nr[k][0] for k in nr if k not in cr]
    only_c = [cr[k][0] for k in cr if k not in nr]
    if only_n or only_c:
        out.append("candidates")
        detail.append(f"only nana: {only_n[:6]}{' ...' if len(only_n) > 6 else ''}; "
                      f"only CC: {only_c[:6]}{' ...' if len(only_c) > 6 else ''}")
    spins = [(nr[k][0], cr[k][0]) for k in nr if k in cr and spin(nr[k][0]) != spin(cr[k][0])]
    if spins:
        out.append("spin")
        detail.append(f"same cells, other spin: {spins[:4]}")
    vals = [(nr[k][0], nr[k][1], cr[k][1]) for k in nr if k in cr and nr[k][1] != cr[k][1]]
    if vals:
        out.append("values")
        deltas = Counter(c - n for _, n, c in vals if n is not None and c is not None)
        detail.append(f"{len(vals)} values differ (CC - nana: {dict(deltas.most_common(5))}), e.g. "
                      + "; ".join(f"{m} nana {n} CC {c}" for m, n, c in vals[:4]))
    norder = [canon(r[0]) for r in ns["ranked"]]
    corder = [canon(r[0]) for r in cs["ranked"] if r[1] is not None]
    if not only_n and not only_c and not vals and norder != corder:
        out.append("ties")
        detail.append("equal values ranked in another order")
    # same value at the same rank but another weight: rounding in the weight formula
    crank = {k: i for i, k in enumerate(corder)}
    bad = [(r[0], r[-1], cr[k][-1]) for i, (k, r) in enumerate(zip(norder, ns["ranked"]))
           if k in cr and crank[k] == i and r[1] == cr[k][1] and r[-1] != cr[k][-1]]
    if bad:
        out.append("weights")
        detail.append(f"{len(bad)} weights differ for equal values at equal ranks (rounding), e.g. {bad[:3]}")
    if abs(ns["u"] - cs["u"]) > 1e-15:
        out.append("draws")
        detail.append(f"draws out of step: nana u={ns['u']} CC u={cs['u']}")
    npick = canon(ns["ranked"][ns["pick"]][0])
    cpick = canon(cs["ranked"][cs["pick"]][0])
    if npick != cpick:
        out.append("pick")
        detail.append(f"picked nana {ns['ranked'][ns['pick']][0]} vs CC {cs['ranked'][cs['pick']][0]}")
    return out, "; ".join(detail)


def walk_trace(nd, cd, show):
    """the first step where the two searches differ at all, and the first where they part"""
    nt = nana_thinks(nd)
    ct = cd["trace"][1:]  # Cold Clear's first think expands the root, which nana starts with
    first_diff = None
    parted = None
    lines = []
    for i in range(min(len(nt), len(ct))):
        if not ct[i]["expanded"]:
            parted = parted or (i + 1, 0, ["cc-no-leaf"], "Cold Clear found no leaf to expand on this think")
            break
        ns, cs = nt[i], ct[i]["steps"]
        for j in range(max(len(ns), len(cs))):
            if j >= len(ns) or j >= len(cs):
                longer = "nana" if len(ns) > len(cs) else "CC"
                why = (f"path lengths differ: nana {len(ns)} steps, CC {len(cs)} "
                       f"({longer} went on below a node the other treats as a leaf)")
                parted = (i + 1, j, ["depth"], why)
                first_diff = first_diff or parted
                break
            kinds, detail = compare_step(ns[j], cs[j])
            if kinds and first_diff is None:
                first_diff = (i + 1, j, kinds, detail)
                if show:
                    lines.append(f"    nana ranked: {fmt_ranked(ns[j]['ranked'], show)}")
                    lines.append(f"    CC   ranked: {fmt_ranked([r for r in cs[j]['ranked'] if r[1] is not None], show)}")
            if "pick" in kinds or "speculation" in kinds or "draws" in kinds:
                parted = (i + 1, j, kinds, detail)
                break
        if parted:
            break
    return first_diff, parted, len(nt), len(ct), lines


def parents(tree):
    count = Counter()
    for node in tree.nodes.values():
        for e in node["edges"].values():
            if e["child"]:
                count[e["child"]] += 1
    return count


def descend(nt, ct, nid, cid, k, npar, cpar, out, cause, depth=0):
    """follow one edge whose values differ down the trees to where the difference starts"""
    ne, ce = nt.nodes[nid]["edges"][k], ct.nodes[cid]["edges"][k]
    line = f"{'  ' * depth}{ne['name']}: nana {ne['value']}, CC {ce['value']}"
    nc, cc = ne["child"], ce["child"]
    if cc and cpar[cc] > 1:
        line += f" [CC child has {cpar[cc]} parents]"
    if nc and npar[nc] > 1:
        line += f" [nana child has {npar[nc]} parents]"
    if not nc and not cc:
        out.append(line + f" - leaf eval differs (move reward nana {ne['reward']}, CC {ce['reward']})")
        cause.add("leaf eval")
        return
    if not nc:
        why = " (CC linked the edge to a node another path expanded)" if cpar[cc] > 1 else ""
        out.append(line + " - CC's child is expanded, nana's edge still has its static eval" + why)
        cause.add("transposition link" if why else "expanded in CC only")
        return
    if not cc:
        out.append(line + " - nana's child is expanded, CC's isn't")
        cause.add("expanded in nana only")
        return
    nn, cn = nt.nodes[nc], ct.nodes[cc]
    if nn["value"] is not None and ne["value"] is not None and abs(ne["reward"] + nn["value"] - ne["value"]) > 1:
        shared = npar[nc] > 1
        out.append(line + f" - nana's edge is stale: its child is worth {nn['value']} now"
                   + (", changed through another parent (CC backs a node up to every parent, nana only along the "
                      "path that changed it)" if shared else ""))
        cause.add("transposition backup" if shared else "stale edge")
        return
    out.append(line)
    best_n = max((e for e in nn["edges"].values() if e["value"] is not None), key=lambda e: e["value"], default=None)
    best_c = max((e for e in cn["edges"].values() if e["value"] is not None), key=lambda e: e["value"], default=None)
    for best, other, side in ((best_c, nn, "CC"), (best_n, cn, "nana")):
        if best and canon(best["name"]) not in other["edges"]:
            out.append(f"{'  ' * (depth + 1)}{side}'s best reply {best['name']} ({best['value']}) isn't in the other's moves")
            cause.add("movegen")
            return
    diff = [kk for kk in nn["edges"] if kk in cn["edges"] and nn["edges"][kk]["value"] != cn["edges"][kk]["value"]]
    if not diff:
        out.append(f"{'  ' * (depth + 1)}children agree; the node values differ (dead moves?)")
        cause.add("node value")
        return
    top = max(diff, key=lambda kk: max(v for v in (nn["edges"][kk]["value"], cn["edges"][kk]["value"]) if v is not None))
    descend(nt, ct, nc, cc, top, npar, cpar, out, cause, depth + 1)


def explain_parting(nd, cd, snap_nd, snap_cd, think, depth):
    """the differing edges where the searches part, followed down the trees as they stood then
    (snap_*: run to a budget of `think`; nana_cctree --sims t leaves the t-th rollout's backup
    undone, so its values are those the t-th select saw). The path comes from the full runs' traces"""
    nt, ct = Tree(snap_nd), Tree(snap_cd)
    ns = nana_thinks(nd)[think - 1]
    cs = cd["trace"][think]["steps"]
    cid = ct.root
    for j in range(depth):
        cid = ct.nodes[cid]["edges"][canon(cs[j]["ranked"][cs[j]["pick"]][0])]["child"]
        if cid is None:
            return [], set()
    nid = ns[depth]["node"]
    cr = {canon(r[0]): r for r in cs[depth]["ranked"] if r[1] is not None}
    npar, cpar = parents(nt), parents(ct)
    out, cause = [], set()
    followed = 0
    for r in ns[depth]["ranked"]:
        k = canon(r[0])
        if k in cr and cr[k][1] != r[1] and k in nt.nodes[nid]["edges"] and k in ct.nodes[cid]["edges"]:
            descend(nt, ct, nid, cid, k, npar, cpar, out, cause)
            followed += 1
            if followed == 8:
                break
    return out, cause


def walk_trees(nt, ct):
    """compare the parts of the trees both searches expanded, node pair by node pair"""
    stats = Counter()
    movegen = []
    static = defaultdict(list)
    backup = []
    seen = set()
    todo = [(nt.root, ct.root, 0)]
    while todo:
        n, c, depth = todo.pop()
        if (n, c) in seen:
            continue
        seen.add((n, c))
        nn, cn = nt.nodes[n], ct.nodes[c]
        if not cn["expanded"]:
            continue
        stats["common nodes"] += 1
        ne, ce = nn["edges"], cn["edges"]
        only_n = [ne[k]["name"] + (" (tops out)" if ne[k]["game_over"] else "") for k in ne if k not in ce]
        only_c = [ce[k]["name"] for k in ce if k not in ne]
        if only_n or only_c:
            stats["nodes with other moves"] += 1
            movegen.append((depth, only_n, only_c))
        for k in ne:
            if k not in ce:
                continue
            a, b = ne[k], ce[k]
            if a["child"] and b["child"]:
                if a["value"] != b["value"]:
                    stats["backed-up values differ"] += 1
                    backup.append((depth, a["name"], a["value"], b["value"]))
                todo.append((a["child"], b["child"], depth + 1))
            elif not a["child"] and not b["child"]:
                stats["leaf edges compared"] += 1
                if a["value"] != b["value"] and a["value"] is not None and b["value"] is not None:
                    stats["leaf values differ"] += 1
                    d_acc = b["reward"] - a["reward"]
                    d_board = (b["value"] - a["value"]) - d_acc
                    static[(a["name"].split()[0], d_acc, d_board)].append((a["name"], b["name"]))
            else:
                stats["expanded in one only"] += 1
    return stats, movegen, static, backup


def stale_edges(nt):
    """nana edges whose value isn't their move's reward plus the child's value"""
    out = []
    for nid, node in nt.nodes.items():
        for e in node["edges"].values():
            child = nt.nodes.get(e["child"]) if e["child"] else None
            if child is None or child["value"] is None or e["value"] is None:
                continue
            want = e["reward"] + child["value"]
            if abs(want - e["value"]) > 1:
                out.append((nid, e["name"], e["value"], want))
    return out


def report(seed, nd, cd, show, snapshot=None):
    nt, ct = Tree(nd), Tree(cd)
    print(f"== seed {seed}: nana {nd['rollouts']} sims (load factor {nd['load_factor']}), "
          f"Cold Clear {cd['thinks']} thinks{'' if cd['known'] == len(nd['position']['queue']) else ', preview only'}")
    incoming = sum(nd["position"].get("meter", [])) + sum(nd["position"].get("buffered", []))
    if incoming:
        print(f"  incoming garbage {nd['position']['meter']} (+{nd['position']['buffered']} delayed): nana's search "
              f"lands it and its eval counts it (NANA_EVAL_TANK); Cold Clear's can't see it (--clear-meter drops it)")
    agree = nd["best"] is not None and cd["best"] is not None and canon(nd["best"]) == canon(cd["best"])
    print(f"  answer: nana {nd['best']}, CC {cd['best']}{'  (same)' if agree else '  (DIFFERENT)'}")
    for name, t in (("nana", nt), ("CC  ", ct)):
        top = ", ".join(f"{e['name']} {e['value']}" for e in t.best_edges(t.root, 4))
        print(f"  {name} root: {top}")

    first, parted, n_n, n_c, lines = walk_trace(nd, cd, show)
    if first:
        think, step, kinds, detail = first
        print(f"  first difference: think {think}, depth {step}: {'+'.join(kinds)}")
        print(f"    {detail}")
        for line in lines:
            print(line)
    else:
        print(f"  selections identical through all {min(n_n, n_c)} thinks")
    if parted and parted != first:
        think, step, kinds, detail = parted
        print(f"  searches part: think {think}, depth {step}: {'+'.join(kinds)}")
        print(f"    {detail}")
    if parted and "values" in parted[2] and snapshot:
        think, step = parted[0], parted[1]
        lines, cause = explain_parting(nd, cd, *snapshot(think), think, step)
        print(f"  at think {think}, followed down (CC - nana value differences):")
        for line in lines:
            print(f"    {line}")
        parted = (think, step, parted[2] + sorted(cause), parted[3])

    stats, movegen, static, backup = walk_trees(nt, ct)
    print("  trees: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    for depth, only_n, only_c in movegen[:3]:
        print(f"    depth {depth}: only nana {only_n[:5]}, only CC {only_c[:5]}")
    if static:
        print("  leaf value differences by piece, move-reward delta, board-value delta (CC - nana):")
        for (piece, d_acc, d_board), moves in sorted(static.items(), key=lambda kv: -len(kv[1]))[:show]:
            print(f"    {piece} acc {d_acc:+d} board {d_board:+d}: {len(moves)} edges, e.g. nana {moves[0][0]} / CC {moves[0][1]}")
    for depth, name, a, b in backup[:3]:
        print(f"    backed-up value differs at depth {depth}: {name} nana {a} CC {b}")
    stale = stale_edges(nt)
    if stale:
        print(f"  nana edges not matching their child (stale or unpropagated backups): {len(stale)}, e.g. "
              + "; ".join(f"{name} has {v}, child says {w}" for _, name, v, w in stale[:3]))
    return {
        "incoming": incoming,
        "agree": agree,
        "first": first[2] if first else [],
        "parted": parted[2] if parted else [],
        "movegen": bool(movegen),
        "static": sum(len(v) for v in static.values()),
        "stale": len(stale),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--count", type=int, default=1, help="positions: seeds S, S+1, ...")
    ap.add_argument("--pieces", type=int, default=10, help="greedy moves played before the position")
    ap.add_argument("--position", help="a position file (nana_eval_dump format) instead of seeds")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--lockstep-seed", type=int, default=1)
    ap.add_argument("--speculate", action="store_true", help="give Cold Clear only the preview")
    ap.add_argument("--canonical", action="store_true",
                    help="both rank ties by occupied cells, nana weighs in integer units (stays in step past ties)")
    ap.add_argument("--clear-meter", action="store_true",
                    help="drop the position's incoming garbage (nana lands it and its eval counts it; Cold Clear can't see it)")
    ap.add_argument("--no-time", action="store_true",
                    help="zero the eval terms that read a move's input time (move_time, jeopardy) on both sides")
    ap.add_argument("--nana", default=str(DEFAULT_NANA))
    ap.add_argument("--cc", default=str(DEFAULT_CC))
    ap.add_argument("--out", help="keep both dumps here")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    work = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="cc_tree_"))
    work.mkdir(parents=True, exist_ok=True)
    config = work / "cc.json"
    settings = {"eval": "cc"}
    if args.no_time:
        settings.update(cc_move_time=0, cc_jeopardy=0)
    config.write_text(json.dumps(settings))
    env = dict(os.environ, NANA_CONFIG=str(config))

    def run_nana(seed, sims):
        cmd = [args.nana, "--sims", sims, "--lockstep-seed", args.lockstep_seed]
        cmd += ["--canonical"] if args.canonical else []
        cmd += ["--clear-meter"] if args.clear_meter else []
        cmd += ["--position", args.position] if args.position else ["--seed", seed, "--pieces", args.pieces]
        return run(cmd, env)

    def run_cc(npath, preview, thinks):
        cmd = [args.cc, "--position", npath, "--thinks", thinks, "--lockstep-seed", args.lockstep_seed]
        cmd += ["--known", preview] if args.speculate else []
        cmd += ["--no-time"] if args.no_time else []
        cmd += ["--canonical"] if args.canonical else []
        return run(cmd)

    results = []
    for i in range(1 if args.position else args.count):
        seed = args.seed + i
        nd = run_nana(seed, args.sims)
        npath = work / f"nana-{seed}.json"
        npath.write_text(json.dumps(nd))
        preview = nd["position"]["preview"]
        cd = run_cc(npath, preview, len(nana_thinks(nd)) + 1)
        (work / f"cc-{seed}.json").write_text(json.dumps(cd))
        # both searches replay exactly, so a budget of t shows the trees the t-th select saw
        snapshot = lambda t, seed=seed, npath=npath, preview=preview: (run_nana(seed, t), run_cc(npath, preview, t))
        results.append(report(seed, nd, cd, args.show, snapshot))
        print()

    if len(results) > 1:
        print(f"== {len(results)} positions")
        print(f"  same answer: {sum(r['agree'] for r in results)}")
        print(f"  positions with incoming garbage: {sum(r['incoming'] > 0 for r in results)}")
        print("  first difference: " + ", ".join(f"{k} {v}" for k, v in
              Counter("+".join(r["first"]) or "none" for r in results).most_common()))
        print("  searches part on: " + ", ".join(f"{k} {v}" for k, v in
              Counter("+".join(r["parted"]) or "never" for r in results).most_common()))
        print(f"  positions with movegen differences: {sum(r['movegen'] for r in results)}")
        print(f"  positions with leaf value differences: {sum(r['static'] > 0 for r in results)}")
        print(f"  positions with stale nana edges: {sum(r['stale'] > 0 for r in results)}")
    print(f"dumps in {work}")


if __name__ == "__main__":
    main()
