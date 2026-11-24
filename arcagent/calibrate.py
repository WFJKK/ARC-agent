"""Choosing how much a model's copies are changed (measure.copy_of): the share of the weights and the size.

    python -m arcagent.calibrate --model models/lin-a --network lin-a-s0 [--lam 1e-6] [--ratio 4] [--out file.json]

A scoring on copies stops a number written into an estimator only if the copies' accuracies differ enough: the best
such number is the copies' mean accuracy, and its mean squared error is their variance. The rule here:

- share: 1% of the weights, and at least 4 of them on average (so a network of 10 weights has 40% changed);
- size: the smallest on a grid at which the variance of the copies' accuracies is at least `ratio` times what the
  best plain sampling loses under the reward, 2 sqrt(lambda s v), v the copies' mean A (1 - A) and s the price of
  one sampled input in forward passes. It must hold at the next size of the grid as well, since a variance
  measured on a hundred copies is itself uncertain.

A coarse pass over the sizes finds where that happens, a finer pass with more copies settles it. The copies'
accuracies say how often the network gives the task's answer, so what this prints belongs in withheld/, and only
share and size go into a run config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from . import measure
from .networks import load_model

COARSE = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
SEED = 20261124


def share_for(weights: dict) -> float:
    n = sum(int((np.asarray(w) != 0).sum()) for w in weights.values())
    return min(1.0, max(0.01, 4.0 / max(n, 1)))


def row(model, weights: dict, share: float, size: float, copies: int, inputs: int, lam: float, s: float, tag: int) -> dict:
    fw, an = model.module().forward, model.task_module().answer
    chunk = min(model.effective_chunk(measure.CHUNK), inputs)
    acc = []
    for j in range(copies):
        w = measure.copy_of(weights, share, size, np.random.default_rng(measure.derive(SEED, f"calibrate {tag} {size}", j)))
        acc.append(measure.accuracy(fw, an, w, model.data_set, model.input_space, inputs, SEED, chunk)["accuracy"])
    a = np.array(acc)
    v = float(np.mean(a * (1 - a)))
    sampling = 2 * math.sqrt(lam * s * v)
    return {"size": size, "copies": copies, "mean": float(a.mean()), "sd": float(a.std()), "q10": float(np.quantile(a, 0.1)),
            "median": float(np.median(a)), "q90": float(np.quantile(a, 0.9)), "variance": float(a.var()),
            "best_sampling_loss": sampling, "ratio": float(a.var() / sampling) if sampling > 0 else 0.0}


def calibrate(model_path: str, network: str, lam: float = 1e-6, ratio: float = 4.0, inputs: int = 32768,
              coarse_copies: int = 32, fine_copies: int = 128, out=print) -> dict:
    m = load_model(model_path)
    w = m.load_weights(network)
    share = share_for(w)
    s = m.sample_ops(w) / m.forward_ops(w)
    n = sum(int((np.asarray(x) != 0).sum()) for x in w.values())
    out(f"{network}: {n:,} weights, share {share:.4g} ({share * n:.1f} changed on average)")

    def show(r):
        out(f"   size {r['size']:6.3f}: mean {r['mean']:.4f}  sd {100 * r['sd']:5.2f} pp  10% {r['q10']:.3f}  "
            f"median {r['median']:.3f}  90% {r['q90']:.3f}   variance / best sampling's loss {r['ratio']:6.2f}")

    coarse, hit = [], None
    for size in COARSE:
        coarse.append(row(m, w, share, size, coarse_copies, inputs, lam, s, 0))
        show(coarse[-1])
        if coarse[-1]["ratio"] >= ratio:
            hit = len(coarse) - 1
            break
    result = {"network": network, "weights": n, "share": share, "lam": lam, "ratio": ratio, "inputs": inputs,
              "coarse": coarse, "fine": [], "size": None}
    if hit is None:
        out(f"   no size up to {COARSE[-1]} spreads the copies' accuracies enough")
        return result
    lo = COARSE[hit - 1] if hit else COARSE[0] / 2
    grid = [float(f"{x:.3g}") for x in np.geomspace(lo, COARSE[hit] * (COARSE[hit] / lo) ** 0.25, 6)[1:]]
    fine = [row(m, w, share, size, fine_copies, inputs, lam, s, 1) for size in grid]
    for r in fine:
        show(r)
    result["fine"] = fine
    for a, b in zip(fine, fine[1:]):
        if a["ratio"] >= ratio and b["ratio"] >= ratio:
            result["size"] = a["size"]
            result["at_size"] = a
            break
    if result["size"] is None:
        out("   the finer pass does not settle it; a larger size or more copies are needed")
    else:
        out(f"   size {result['size']}: the copies' accuracies have a standard deviation of {100 * result['at_size']['sd']:.2f} points "
            f"around {result['at_size']['mean']:.4f}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--inputs", type=int, default=32768)
    ap.add_argument("--out", help="write the result as JSON here")
    args = ap.parse_args(argv)
    result = calibrate(args.model, args.network, args.lam, args.ratio, args.inputs)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0 if result["size"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
