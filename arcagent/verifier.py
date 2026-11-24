"""The verifier: measures an estimator's error and its compute on one network.

    python -m arcagent.verifier --model models/rnn-g --network rnn-g-s0 --estimator e.json [--runs 4096] [--lam 1e-6]

It reads an estimator as JSON data and interprets it (lang.py); nothing of an estimator is ever executed as code.
It measures in three steps:

1. **The network's accuracy, A.** The network is run on reference_inputs inputs drawn from the data set with
   reference_seed, and the share of them on which it gives the task's answer is counted (networks.py,
   Model.accuracy). The network and the task run through the same interpreter as an estimator. A is never
   shown to the agent.
2. **The estimates.** The estimator is run `runs` times, run k with the seed [seed, k], where seed is drawn
   afresh for each scoring from a source nobody chose. Each run gives an estimate and its operations.
3. **The parts and the reward.** mse is the mean over runs of (estimate - A)^2, cost the mean operations of a run
   over F, the operations of one forward pass. The reward is rewards.py's function of these.

The receipt also carries a benchmark, which the agent is not shown: plain sampling at the same cost. One more
sampled input costs plain sampling the draws, a forward pass, the task's answer and a comparison (S operations,
networks.py, Model.sample_ops), so at the estimator's mean cost C sampling can afford n = C F / S inputs, and its
mean squared error is A (1 - A) / n. The best plain sampling can do under r_lambda takes n = sqrt(A (1 - A) /
(lambda s)) inputs, s = S / F, and scores -2 sqrt(lambda s A (1 - A)).

**On copies.** Scored like this, the best estimator is a number: the agent's own code may run the network as
often as it likes at no charge, so the accuracy, measured there and written in as a constant, has no error and no
cost. When the config has a `copies` block (verifier.copies: count, runs, reference_inputs, share, size), a
scoring is therefore made on copies of the network instead (measure.py): `count` copies, each with a few weights
changed at random, drawn afresh from the scoring's seed. Each copy's accuracy is measured on reference_inputs
inputs, the estimator is run `runs` times on each copy, and a run's error is its estimate less the accuracy of the
copy it ran on. The parts are means over all runs on all copies, and the reward is the same function of them. The
benchmark is plain sampling on the same copies (A (1 - A) averaged over them), and two constants: the best one
there could be for these copies, and the unchanged network's own accuracy written in. score(..., copies=False)
scores on the network itself whatever the config says: held-out networks are scored so, and the locked estimator
once, for the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from . import lang, measure, rewards
from .networks import Model, load_model

DEFAULTS = {"runs": 4096, "reference_inputs": 2097152, "reference_seed": 20261124, "reference_chunk": 32768,
            "time_limit_s": 600.0, "copies": None}
ENTRY_DEFAULTS = {"max_nodes": 400, "max_const_scalars": 256}
REFERENCE_KEYS = ("n", "seed", "chunk")


def sampling_benchmark(accuracy: float, mse: float, cost: float, sample_cost: float, lam: float | None,
                       variance: float | None = None) -> dict:
    """Plain sampling at the estimator's cost, and the best plain sampling under the reward (see the module's text).
    Costs in forward passes; sample_cost is what one more sampled input costs. variance: the variance of one
    sampled input if it is not accuracy (1 - accuracy): on copies, the mean of that over the copies."""
    var = accuracy * (1.0 - accuracy) if variance is None else float(variance)
    n = cost / sample_cost
    out = {"sample_cost": sample_cost, "variance": var, "samples_at_cost": n}
    if n >= 1:
        mc = var / n
        out.update(mse_at_cost=mc, rms_error_at_cost=math.sqrt(mc),
                   mse_reduction=(1.0 - mse / mc) if mc > 0 else None)
    else:
        out.update(mse_at_cost=None, rms_error_at_cost=None, mse_reduction=None)
    if lam is not None and lam > 0 and var > 0:
        best = -2.0 * math.sqrt(lam * sample_cost * var)
        out.update(best_samples=math.sqrt(var / (lam * sample_cost)), best_reward=best)
    return out


class Verifier:
    def __init__(self, model: Model, network_id: str, cfg: dict | None = None, reference: dict | None = None):
        cfg = cfg or {}
        self.model, self.network_id = model, network_id
        self.cfg = {**DEFAULTS, **(cfg.get("verifier") or {})}
        self.limits = {**ENTRY_DEFAULTS, **(cfg.get("entry") or {})}
        self.reward_cfg = dict(cfg.get("reward") or {"name": "r_lambda"})
        rewards.check(self.reward_cfg)
        self.copies = measure.copies_spec(self.cfg.get("copies"))  # None: scoring is on the network itself
        self.weights = model.load_weights(network_id)
        self.F = model.forward_ops(self.weights)
        self.sample_ops = model.sample_ops(self.weights)
        self._reference = None
        if reference is not None:
            self.set_reference(reference)

    def wanted(self) -> dict:
        """The reference measurement this verifier's config asks for."""
        return {"n": int(self.cfg["reference_inputs"]), "seed": int(self.cfg["reference_seed"]), "chunk": self.chunk()}

    def set_reference(self, ref: dict) -> None:
        """Take a measurement made before (a run folder keeps them), if it is the one this config asks for."""
        if {k: ref.get(k) for k in REFERENCE_KEYS} != self.wanted() or ref.get("network") not in (None, self.network_id):
            raise ValueError(f"the reference measurement {ref} is not the one the config asks for: {self.wanted()}")
        self._reference = dict(ref)

    def reference(self) -> dict:
        """The network's accuracy as the verifier measures it: {"accuracy", "correct", "n", "seed", "chunk", "method",
        "inputs", "standard_error", "network"}; method exact when every input of the data set was run."""
        if self._reference is None:
            w = self.wanted()
            r = self.model.accuracy(self.weights, w["n"], w["seed"], w["chunk"])
            self._reference = {**r, "network": self.network_id}
        return self._reference

    def chunk(self) -> int:
        return self.model.effective_chunk(int(self.cfg["reference_chunk"]))

    def score(self, entry, seed: int | None = None, runs: int | None = None, keep_runs: bool = False,
              copies: bool | None = None, count: int | None = None) -> dict:
        """The receipt: status scored or refused (with the reason), the estimator's hash and size, the seed and the
        number of runs, the parts measured and the reward under the run's reward function, and the benchmark.

        copies: score on copies of the network (the config's default when it has a copies block) or, with False,
        on the network itself. On copies, `runs` is the runs on each copy and `count` the number of copies.
        keep_runs: also every run's estimate, operations and draws, under "runs_data"."""
        t0 = time.monotonic()
        on_copies = self.copies is not None if copies is None else bool(copies)
        if on_copies and self.copies is None:
            raise ValueError("this verifier's config has no copies block to score on copies with")
        if on_copies:
            runs, count = int(runs or self.copies["runs"]), int(count or self.copies["count"])
        else:
            runs, count = int(runs or self.cfg["runs"]), None
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big")
        receipt = {"status": "refused", "reason": None, "network": self.network_id, "entry_sha256": None, "size": None,
                   "seed": int(seed), "runs": runs * (count or 1), "scored_on": "copies" if on_copies else "network",
                   "parts": None, "reward": None, "prices": lang.PRICES, "format": lang.FORMAT}
        try:
            receipt["size"] = lang.check_structure(entry, self.limits["max_nodes"], self.limits["max_const_scalars"])
            receipt["entry_sha256"] = lang.entry_hash(entry)
            if on_copies:
                raw = measure.on_copies(entry, self.weights, self.model.module().forward, self.model.task_module().answer,
                                        self.model.data_set, self.model.input_space, self.copies, int(seed), count, runs,
                                        self.chunk(), float(self.cfg["time_limit_s"]))
            else:
                r = lang.run_estimator(entry, self.weights, runs, int(seed),
                                       deadline=time.monotonic() + float(self.cfg["time_limit_s"]))
        except lang.EntryError as exc:
            receipt["reason"] = str(exc)[:1000]
            receipt["seconds"] = round(time.monotonic() - t0, 3)
            return receipt
        except Exception as exc:  # not an estimator the interpreter could even fail on properly: refused, and marked
            # The reason goes back to the agent, so it names the kind of failure only; the detail stays in the receipt.
            receipt["reason"] = f"the verifier could not read or run this estimator ({type(exc).__name__})"
            receipt["verifier_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            receipt["seconds"] = round(time.monotonic() - t0, 3)
            return receipt
        lam, s = rewards.lam_of(self.reward_cfg), self.sample_ops / self.F
        if on_copies:
            A = raw["accuracy"]
            parts = measure.copies_parts(raw, self.F)
            bench = sampling_benchmark(parts["accuracy"], parts["mse"], parts["cost"], s, lam, variance=float(np.mean(A * (1 - A))))
            # what a number written into the estimator could score on these copies: the best one there is for them
            # (their mean accuracy), and the unchanged network's own accuracy
            bench["constant_best"] = -float(A.var())
            bench["constant_unchanged"] = -float(np.mean((self.reference()["accuracy"] - A) ** 2))
            receipt["reference"] = {**raw["reference"], "copies": int(len(A))}
            receipt["copies"] = {**{k: self.copies[k] for k in ("share", "size", "reference_inputs")}, "count": int(len(A)),
                                 "runs": runs, "accuracy": A.tolist(), "estimate_mean": raw["estimates"].mean(axis=1).tolist(),
                                 "seeds": raw["seeds"]}
            kept = {"estimates": raw["estimates"], "ops": raw["ops"], "draws": raw["draws"], "accuracy": A}
        else:
            ref = self.reference()
            parts = self.parts(r, ref["accuracy"])
            bench = sampling_benchmark(ref["accuracy"], parts["mse"], parts["cost"], s, lam)
            receipt["reference"] = {k: ref[k] for k in ("n", "seed", "chunk", "standard_error")}
            kept = {"estimates": r["estimates"], "ops": r["ops"], "draws": r["draws"]}
        out = rewards.compute(parts, self.reward_cfg)
        receipt.update(status="scored", parts=parts, reward=out["reward"],
                       reward_detail={k: v for k, v in out.items() if k != "reward"}, benchmark=bench)
        if keep_runs:
            receipt["runs_data"] = {k: v.tolist() for k, v in kept.items()}
        receipt["seconds"] = round(time.monotonic() - t0, 3)
        return receipt

    def parts(self, r: dict, accuracy: float) -> dict:
        est, ops = r["estimates"], r["ops"].astype(np.float64)
        err = est - accuracy
        mse = float(np.mean(err**2))
        return {
            "accuracy": float(accuracy), "runs": int(len(est)), "random": bool(r["random"]),
            "estimate_mean": float(est.mean()), "estimate_std": float(est.std()) if est.min() < est.max() else 0.0,
            "estimate_min": float(est.min()),
            "estimate_max": float(est.max()), "mse": mse, "rms_error": math.sqrt(mse), "bias": float(err.mean()),
            "cost": float(ops.mean() / self.F), "cost_min": float(ops.min() / self.F), "cost_max": float(ops.max() / self.F),
            "ops_mean": float(ops.mean()), "draws_mean": float(r["draws"].mean()), "forward_ops": int(self.F),
        }


def agent_view(receipt: dict) -> dict:
    """What the agent is told of a scoring: the reward and what it can know of its own estimator, but not the
    accuracy the verifier measured (the network's, or the copies'), the bias of the estimates or the benchmark,
    which would hand it that accuracy. On copies the spread of the estimates comes twice: within a copy and
    between copies."""
    out = {"status": receipt["status"], "reason": receipt["reason"], "reward": receipt["reward"], "size": receipt["size"],
           "runs": receipt["runs"]}
    p = receipt.get("parts")
    if p:
        out.update(mean_squared_error=p["mse"], rms_error=p["rms_error"], cost=p["cost"], cost_min=p["cost_min"],
                   cost_max=p["cost_max"], estimate_mean=p["estimate_mean"], estimate_std=p["estimate_std"],
                   random_draws_per_run=p["draws_mean"], uses_random_draws=p["random"])
        if "copies" in p:
            out.update(copies=p["copies"], runs_per_copy=p["runs_per_copy"], estimate_std_between_copies=p["estimate_between"])
    return out


def describe(receipt: dict) -> str:
    """One line for logs and reports."""
    if receipt["status"] != "scored":
        return f"refused: {receipt['reason']}"
    p = receipt["parts"]
    over = f"over {p['runs']} runs on {p['copies']} copies" if "copies" in p else f"over {p['runs']} runs"
    line = (f"reward {receipt['reward']:.6e}  rms error {100 * p['rms_error']:.3f} pp  bias {100 * p['bias']:+.3f} pp  "
            f"cost {p['cost']:.2f} F  {over}")
    b = receipt.get("benchmark") or {}
    if b.get("mse_reduction") is not None:
        red = 100 * b["mse_reduction"]
        line += f"  (mse {abs(red):.1f}% {'below' if red >= 0 else 'above'} sampling's at that cost)"
    return line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--estimator", required=True, help="an estimator as a JSON file")
    ap.add_argument("--runs", type=int, default=DEFAULTS["runs"])
    ap.add_argument("--lam", type=float, default=1e-6, help="lambda of r_lambda: the price of one forward pass")
    ap.add_argument("--seed", type=int, help="the scoring's seed (default: drawn afresh)")
    ap.add_argument("--reference-inputs", type=int, default=DEFAULTS["reference_inputs"])
    args = ap.parse_args(argv)
    v = Verifier(load_model(args.model), args.network, {"reward": {"name": "r_lambda", "lam": args.lam},
                                                       "verifier": {"reference_inputs": args.reference_inputs}})
    with open(args.estimator) as f:
        receipt = v.score(json.load(f), seed=args.seed, runs=args.runs)
    print(json.dumps(receipt, indent=1))
    print(describe(receipt), file=sys.stderr)
    return 0 if receipt["status"] == "scored" else 1


if __name__ == "__main__":
    sys.exit(main())
