"""The report of a run, from its records alone; and the reference estimators of a model, with no run at all.

    python -m arcagent.report --run <run folder>                writes report.md and report.json again
    python -m arcagent.report --model models/rnn-g --network rnn-g-s0 [--runs 4096] [--lam 1e-6]
                                                              the reference estimators and plain sampling there

A run's report gives the network's accuracy as the verifier measured it, every submission with its parts, the
locked estimator scored again with a fresh seed, the locked estimator on every held-out network, the reference
estimators and plain sampling under the same conditions, and the agent's closing notes. It marks the submissions on
the frontier of mean squared error and cost, which needs no lambda.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .util import atomic_write, read_json, read_jsonl, write_json


def _pp(x) -> str:
    return "" if x is None else f"{100 * x:.3f}"


def _e(x) -> str:
    return "" if x is None else f"{x:.4e}"


def _c(x) -> str:
    return "" if x is None else f"{x:.2f}"


def _reward(cfg: dict) -> str:
    params = ", ".join(f"{'lambda' if k == 'lam' else k} = {v:g}" for k, v in sorted(cfg.items()) if k != "name")
    return f"`{cfg.get('name', 'r_lambda')}`" + (f", {params}" if params else "")


def frontier(rows: list) -> set:
    """The rows no other row beats on both mean squared error and cost (lower is better on both)."""
    out = set()
    for i, a in enumerate(rows):
        beaten = any(b["mse"] <= a["mse"] and b["cost"] <= a["cost"] and (b["mse"] < a["mse"] or b["cost"] < a["cost"])
                     for j, b in enumerate(rows) if j != i)
        if not beaten:
            out.add(i)
    return out


def _row(label: str, r: dict) -> list:
    if r.get("status") != "scored":
        return [label, f"refused: {str(r.get('reason'))[:80]}", "", "", "", "", "", ""]
    p, b = r["parts"], r.get("benchmark") or {}
    red = b.get("mse_reduction")
    return [label, _e(r["reward"]), _pp(p["rms_error"]), f"{100 * p['bias']:+.3f}", _pp(p["estimate_std"]), _c(p["cost"]),
            "" if red is None else f"{100 * red:.1f}%", "yes" if p["random"] else "no"]


HEAD = ["", "reward", "rms error (pp)", "bias (pp)", "spread (pp)", "cost (forward passes)",
        "mse below sampling's at that cost", "draws"]


def _table(rows: list) -> list:
    out = ["| " + " | ".join(HEAD) + " |", "|" + " --- |" * len(HEAD)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def gather(path: Path) -> dict:
    from .loop import HELDOUT_SCORES, completed, heldout_networks, scored
    from .session import cached_references

    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    rounds = completed(path)
    subs = scored(rounds)
    held = {}
    for line in read_jsonl(path / HELDOUT_SCORES):
        held.setdefault(line["carried_as"], {})[line["network"]] = line
    return {
        "run": run, "cfg": cfg, "rounds": rounds, "submissions": subs, "heldout_scores": held,
        "references": cached_references(path), "nets": heldout_networks(cfg),
        "lock": read_json(path / "lock.json") if (path / "lock.json").is_file() else None,
        "check": read_json(path / "heldout_check.json") if (path / "heldout_check.json").is_file() else None,
        "reference_scores": read_json(path / "reference_scores.json") if (path / "reference_scores.json").is_file() else None,
    }


def write(path: Path) -> dict:
    path = Path(path)
    d = gather(path)
    run, cfg = d["run"], d["cfg"]
    refs = d["references"]
    net = cfg["network"]
    A = refs.get(net, {}).get("accuracy")
    lines = [f"# Run {cfg['run_id']}", ""]
    lines += [f"- Model `{cfg['model']}`, network `{net}`; agent `{cfg.get('agent_model')}` (effort {cfg.get('effort')}), "
              f"backend {cfg['backend']}; sandbox {run['sandbox']['mode']}.",
              f"- Reward {_reward(cfg['reward'])}; {int(cfg['verifier']['runs']):,} runs per scoring; a forward pass is "
              f"{run['forward_ops']:,} operations, one more sampled input {run['sample_ops']:,} ({run['sample_ops'] / run['forward_ops']:.4f} "
              f"forward passes).",
              f"- The network's accuracy as the verifier measured it: {A:.6f} on {refs[net]['n']:,} inputs (seed {refs[net]['seed']}), "
              f"standard error {100 * refs[net]['standard_error']:.4f} points. The agent is not shown it." if A is not None else
              "- The network's accuracy: not measured.", ""]

    # the submissions
    lines += ["## Submissions", ""]
    rows, frows = [], []
    for s in d["submissions"]:
        hs = d["heldout_scores"].get(s["carried_as"], {})
        ok = [x for x in hs.values() if x["status"] == "scored"]
        mean_held = sum(x["reward"] for x in ok) / len(ok) if ok else None
        rows.append(_row(f"{s['carried_as']} `{s['name']}`", s) + [_e(mean_held)])
        frows.append({"mse": s["parts"]["mse"], "cost": s["parts"]["cost"]})
    on_front = frontier(frows)
    if rows:
        head = HEAD + ["mean reward held out"]
        lines += ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        lines += ["| " + " | ".join(r[:1] + [r[1] + (" *" if i in on_front else "")] + r[2:]) + " |" for i, r in enumerate(rows)]
        lines += ["", "\\* on the frontier of mean squared error and cost. Bias is the mean estimate less the accuracy; spread is "
                  "the standard deviation of the estimates over the runs. Held out: every submission scored on every "
                  f"held-out network with {int((cfg.get('heldout') or {}).get('runs') or 0):,} runs and one fixed seed per "
                  "network; the agent never sees these.", ""]
    else:
        lines += ["No submission was scored.", ""]
    refused = [s for rd in d["rounds"] for s in rd["submissions"] if s["status"] != "scored"]
    if refused:
        lines += [f"Refused: {len(refused)} (" + "; ".join(f"{s.get('name')}: {str(s.get('reason'))[:80]}" for s in refused[:5]) + ")", ""]

    # the lock
    lock = d["lock"]
    if lock and lock.get("locked"):
        lines += ["## The locked estimator", "",
                  f"`{lock['carried_as']}` ({lock['name']}), sha256 {lock['entry_sha256'][:16]}, {lock['size']['nodes']} nodes.", ""]
        lines += _table([_row("as submitted", lock), _row("scored again, fresh seed", lock.get("measured_again") or {})])
        lines.append("")
        check = d["check"]
        if check:
            lines += ["### On the held-out networks", ""]
            head = ["network", "accuracy", "estimate (mean)", "error (pp)", "rms error (pp)", "cost", "reward",
                    "sampling at that cost: rms (pp)", "a constant of the run's accuracy: reward"]
            lines += ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
            for n, r in check["networks"].items():
                An = refs.get(n, {}).get("accuracy")
                if r.get("status") != "scored":
                    lines.append(f"| {n} | {An} | refused: {str(r.get('reason'))[:60]} | | | | | | |")
                    continue
                p, b = r["parts"], r.get("benchmark") or {}
                const = -((A - An) ** 2) if A is not None and An is not None else None
                lines.append(f"| {n} | {An:.6f} | {p['estimate_mean']:.6f} | {100 * p['bias']:+.3f} | {_pp(p['rms_error'])} | "
                             f"{_c(p['cost'])} | {_e(r['reward'])} | {_pp(b.get('rms_error_at_cost'))} | {_e(const)} |")
            ok = [r for r in check["networks"].values() if r.get("status") == "scored"]
            if ok:
                mean = sum(r["reward"] for r in ok) / len(ok)
                lines += ["", f"Mean reward on the held-out networks: {mean:.4e}. The last column is what the run's own accuracy, "
                          "written in as a constant, would score there: the route a reward on one network cannot see.", ""]
    elif lock:
        lines += ["## The locked estimator", "", f"Nothing locked: {lock.get('reason')}.", ""]

    # the references
    rs = d["reference_scores"]
    if rs:
        lines += ["## Reference estimators", "", "The withheld references and plain sampling, under the run's conditions "
                  f"(seed {rs['seed']}).", ""]
        rows = []
        for n, scores in rs["networks"].items():
            for r in scores:
                rows.append(_row(f"{r['name']} on {n}", r))
        lines += _table(rows)
        lines.append("")

    # the agent
    if d["rounds"]:
        last = d["rounds"][-1]
        meta = {}
        try:
            meta = read_json(path / f"round_{last['round']:03d}" / "meta.json")
        except (OSError, ValueError):
            pass
        lines += ["## The agent", ""]
        for rd in d["rounds"]:
            lines.append(f"- Round {rd['round']}: {len(rd['submissions'])} submissions, tools used {rd.get('tools_used')}, "
                         f"best reward after {_e(rd.get('best_reward_after'))}, stop reason {rd.get('stop_reason')}.")
        if meta:
            lines.append(f"- Last round: {meta.get('num_turns')} turns, {meta.get('output_tokens')} output tokens, "
                         f"cost ${meta.get('cost_usd') or 0:.2f}.")
        lines += ["", "### Its closing notes, last round", "", "\n".join("    " + x for x in (last.get("notes") or "(none)").splitlines()), ""]

    text = "\n".join(lines).rstrip() + "\n"
    atomic_write(path / "report.md", text)
    summary = {"run_id": cfg["run_id"], "accuracy": A, "lock": lock, "heldout_check": d["check"],
               "submissions": [{k: s[k] for k in ("carried_as", "name", "reward", "parts", "benchmark")} for s in d["submissions"]]}
    write_json(path / "report.json", summary)
    return summary


def model_references(model_path: str, network: str, runs: int, lam: float, sizes: list, out=print) -> list:
    """The withheld references and plain sampling on one network, with no run."""
    from .networks import load_model
    from .verifier import Verifier

    m = load_model(model_path)
    v = Verifier(m, network, {"reward": {"name": "r_lambda", "lam": lam}, "verifier": {"runs": runs}})
    ref = v.reference()
    A = ref["accuracy"]
    out(f"{network}: accuracy {A:.6f} on {ref['n']:,} inputs (seed {ref['seed']}); forward pass {v.F:,} operations; "
        f"one more sampled input {v.sample_ops / v.F:.4f} forward passes")
    ests = dict(m.reference_entries(v.weights))
    best = max(1, round(math.sqrt(A * (1 - A) / (lam * v.sample_ops / v.F)))) if lam > 0 else None
    for n in sorted(set(sizes + ([best] if best else []))):
        ests[f"plain sampling, {n} inputs"] = m.monte_carlo(v.weights, n)
    rows = []
    for name, est in ests.items():
        r = v.score(est, seed=0)
        rows.append(_row(name, r))
    for line in _table(rows):
        out(line)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="a run folder: write its report again")
    which.add_argument("--model", help="a model folder: score its reference estimators")
    ap.add_argument("--network", help="with --model: the network (default: the first)")
    ap.add_argument("--runs", type=int, default=4096)
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--sizes", default="16,64,256", help="with --model: plain sampling at these numbers of inputs")
    args = ap.parse_args(argv)
    if args.run:
        write(Path(args.run))
        print(Path(args.run) / "report.md")
        return 0
    from .networks import load_model

    network = args.network or load_model(args.model).network_ids()[0]
    model_references(args.model, network, args.runs, args.lam, [int(s) for s in args.sizes.split(",") if s])
    return 0


if __name__ == "__main__":
    sys.exit(main())
