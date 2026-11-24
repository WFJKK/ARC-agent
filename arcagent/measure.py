"""Measuring: a network's accuracy, and an estimator on copies of a network.

The verifier measures with these functions (verifier.py), and this file is placed in the agent's sandbox too, so
that the agent's code can measure exactly as the verifier does, on seeds and copies of its own. Nothing here says
anything about any one network.

**Accuracy.** How often a network gives the task's answer on its data set, with the network and the task run
through the interpreter as an estimator would run them. A data set that can be listed and has no more inputs than
asked for is measured on every input, each weighted by how often it is drawn (method exact); otherwise the network
is run on inputs drawn from the data set (method sample).

**Copies.** A reward that scores an estimator on the one network its author holds cannot tell an estimate computed
from the weights from a number measured beforehand and written in. So an estimator can be scored on copies of the
network instead: the same network with a few of its weights changed at random (copy_of), made afresh for every
scoring from the scoring's own seed. What an estimator computes from the weights follows a copy; a number written
into it does not. Every copy's accuracy is measured as above, the estimator is run on the copy, and the error of
each run is its estimate less that copy's accuracy (on_copies, copies_parts).
"""

from __future__ import annotations

import hashlib
import math
import time

import numpy as np

try:
    from . import lang
except ImportError:  # in a sandbox folder the modules are plain files next to each other
    import lang

CHUNK = 32768  # inputs at once, unless a model says less (config.yaml: max_batch)


# accuracy --------------------------------------------------------------------------------------------
def monte_carlo(forward, answer, weights: dict, data_set, input_space, n: int) -> dict:
    """Plain sampling as an estimator: n inputs drawn afresh in every run, the share on which the network gives
    the task's answer. It costs n times (the draws, a forward pass, the answer and one comparison) plus the sum and
    the division. The estimator names its weights and does not hold them, so it runs on any network of the model."""
    g = lang.Graph(weights, data_set, input_space)
    x = g.inputs(int(n))
    right = forward(g, lang, x) == answer(g, lang, x)
    return g.entry(right.sum() / float(n))


def outputs(forward, answer, weights: dict, data_set, input_space, x, chunk: int = CHUNK) -> tuple:
    """The network's outputs and the task's answers on the inputs x (a numpy array, one row each), computed with
    forward and answer through the interpreter, chunk inputs at a time."""
    x = np.asarray(x)
    got, want = [], []
    for i in range(0, len(x), int(chunk)):
        g = lang.Graph(weights, data_set, input_space)
        xs = g.const(x[i:i + int(chunk)])
        both = lang.stack([forward(g, lang, xs), answer(g, lang, xs)], axis=0)
        v = lang.Evaluator(g.entry(both), weights).run([0])["estimate"]
        got.append(v[0])
        want.append(v[1])
    return np.concatenate(got), np.concatenate(want)


def accuracy(forward, answer, weights: dict, data_set, input_space, n: int, seed: int, chunk: int = CHUNK,
             sampler: dict | None = None) -> dict:
    """How often the network gives the task's answer.

    A data set that can be listed and has at most n inputs is measured exactly, on every input, each weighted by
    how likely it is. Otherwise the network is run on n inputs drawn from the data set with the seed, chunk inputs
    at a time: chunk k is plain sampling of chunk inputs (monte_carlo) run with the seed [seed, k]. `sampler` is
    that estimator if it was built before (it is the same for every network of a model)."""
    n, chunk = int(n), int(chunk)
    total = lang.count_inputs(data_set, input_space)
    if total is not None and total <= n:
        share, correct, groups = 0.0, 0, []
        for x, p in lang.enumerate_inputs(data_set, input_space):
            out, ans = outputs(forward, answer, weights, data_set, input_space, x, chunk)
            k = int((out == ans).sum())
            share += k * p
            correct += k
            groups.append({"inputs": int(len(x)), "correct": k, "probability_each": p})
        return {"accuracy": float(share), "correct": correct, "n": n, "seed": int(seed), "chunk": chunk,
                "method": "exact", "inputs": int(total), "groups": groups, "standard_error": 0.0}
    if n < 1 or chunk < 1 or n % chunk:
        raise ValueError("the inputs must be a positive multiple of the chunk")
    if sampler is None:
        sampler = monte_carlo(forward, answer, weights, data_set, input_space, chunk)
    r = lang.run_estimator(sampler, weights, n // chunk, int(seed))
    correct = int(sum(int(round(e * chunk)) for e in r["estimates"]))
    a = correct / n
    return {"accuracy": a, "correct": correct, "n": n, "seed": int(seed), "chunk": chunk, "method": "sample",
            "inputs": n, "standard_error": math.sqrt(a * (1 - a) / n)}


# copies ----------------------------------------------------------------------------------------------
SPEC_KEYS = ("count", "runs", "reference_inputs", "share", "size")


def copies_spec(spec) -> dict | None:
    """A config's `copies` block, checked: None when scoring is on the network itself (no block, or count 0)."""
    if not spec or not int(spec.get("count") or 0):
        return None
    missing = [k for k in SPEC_KEYS if spec.get(k) is None]
    if missing:
        raise ValueError(f"scoring on copies needs {', '.join(missing)} (verifier.copies in the run config); "
                         "count: 0 scores on the network itself")
    out = {"count": int(spec["count"]), "runs": int(spec["runs"]), "reference_inputs": int(spec["reference_inputs"]),
           "share": float(spec["share"]), "size": float(spec["size"])}
    if out["count"] < 1 or out["runs"] < 1 or out["reference_inputs"] < 1:
        raise ValueError("copies: count, runs and reference_inputs must be at least 1")
    if not 0 < out["share"] <= 1 or not out["size"] > 0:
        raise ValueError("copies: share is a probability above 0, and size is above 0")
    return out


def derive(seed: int, what: str, j: int = 0) -> int:
    """A seed for one part of a scoring, from the scoring's seed: which part (`what`) of which copy (j). Parts get
    seeds that have nothing to do with each other, and all of them follow from the one seed a receipt records."""
    return int.from_bytes(hashlib.sha256(f"{int(seed)}:{what}:{int(j)}".encode()).digest()[:8], "big")


def copy_of(weights: dict, share: float, size: float, rng) -> dict:
    """A copy of a network with a few of its weights changed at random.

    Every weight is changed with probability `share`, each on its own: a normal draw is added to it, with standard
    deviation `size` times the root mean square of the entries of its weight matrix. The other weights stay as they
    are, bit for bit. A weight that is exactly zero is never changed: such a weight is part of how the network is
    built (a slot that holds no token adds nothing). The matrices are gone through in the order of their names,
    each taking its draws from rng: first which weights change, then by how much. Weights are kept in float32, as
    the networks' files keep them. Now and then no weight is changed at all (about one copy in 70 where 4 change
    on average): that copy is the network itself, one among the many of a scoring."""
    out = {}
    for name in sorted(weights):
        w = np.asarray(weights[name], dtype=np.float64)
        rms = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
        changed = rng.random(w.shape) < float(share)
        step = float(size) * rms * rng.standard_normal(w.shape)
        out[name] = np.where(changed & (w != 0.0), w + step, w).astype(np.float32)
    return out


def on_copies(entry: dict, weights: dict, forward, answer, data_set, input_space, spec: dict, seed: int,
              count: int | None = None, runs: int | None = None, chunk: int = CHUNK,
              time_limit_s: float | None = None) -> dict:
    """An estimator on copies of a network: for copy j = 0 .. count - 1,

        its weights     copy_of(weights, share, size, numpy.random.default_rng(derive(seed, "copy", j)))
        its accuracy    accuracy(...) on reference_inputs inputs, drawn with the seed derive(seed, "reference", j)
        the estimator   run `runs` times on the copy's weights, run k with the seed [derive(seed, "runs", j), k]

    Returns the copies' accuracies (count,), every run's estimate, operations and draws (count, runs), whether the
    estimator draws at all, the copies' seeds and how the accuracies were measured. time_limit_s bounds the
    estimator's runs together; measuring the copies' accuracies takes no part of it. An estimator that fails on a
    copy is refused as a whole (lang.EntryError), and the message names the copy."""
    count, runs = int(count or spec["count"]), int(runs or spec["runs"])
    n_ref = int(spec["reference_inputs"])
    total = lang.count_inputs(data_set, input_space)
    exact = total is not None and total <= n_ref
    chunk = min(int(chunk), n_ref)
    sampler = None if exact else monte_carlo(forward, answer, weights, data_set, input_space, chunk)
    left = None if time_limit_s is None else float(time_limit_s)
    acc, est, ops, draws, seeds, random, se = [], [], [], [], [], False, []
    for j in range(count):
        cseed = derive(seed, "copy", j)
        w = copy_of(weights, spec["share"], spec["size"], np.random.default_rng(cseed))
        ref = accuracy(forward, answer, w, data_set, input_space, n_ref, derive(seed, "reference", j), chunk, sampler)
        t = time.monotonic()
        try:
            r = lang.run_estimator(entry, w, runs, derive(seed, "runs", j), deadline=None if left is None else t + left)
        except lang.EntryError as exc:
            raise lang.EntryError(f"on a copy of the network (copy({cseed}) makes it, with the run seeds "
                                  f"[{derive(seed, 'runs', j)}, k]): {exc}") from None
        if left is not None:
            left = max(0.0, left - (time.monotonic() - t))
        acc.append(ref["accuracy"])
        se.append(ref["standard_error"])
        est.append(r["estimates"])
        ops.append(r["ops"])
        draws.append(r["draws"])
        seeds.append(cseed)
        random = random or bool(r["random"])
    return {"accuracy": np.array(acc, dtype=np.float64), "estimates": np.array(est, dtype=np.float64),
            "ops": np.array(ops), "draws": np.array(draws), "random": random, "seeds": seeds,
            "reference": {"method": "exact" if exact else "sample", "inputs": int(total) if exact else n_ref,
                          "chunk": chunk, "standard_error": float(max(se)) if se else 0.0}}


def copies_parts(raw: dict, forward_ops: int) -> dict:
    """What a scoring on copies measured, in the parts every receipt keeps. A run's error is its estimate less the
    accuracy of the copy it ran on; mse, bias and cost are means over all runs on all copies. The spread of the
    estimates is given twice: within a copy (the mean, over the copies, of the standard deviation of its runs) and
    between copies (the standard deviation of the copies' mean estimates). tracking is the correlation between a
    copy's mean estimate and its accuracy: near 1 for an estimator that follows the copies, none for a constant."""
    A, est, ops = raw["accuracy"], raw["estimates"], raw["ops"].astype(np.float64)
    F = float(forward_ops)
    err = est - A[:, None]
    mse = float(np.mean(err**2))
    per = est.mean(axis=1)
    within = np.where(est.max(axis=1) > est.min(axis=1), est.std(axis=1), 0.0)
    moves = bool(per.max() > per.min()) and bool(A.max() > A.min())
    return {
        "accuracy": float(A.mean()), "accuracy_std": float(A.std()), "accuracy_min": float(A.min()),
        "accuracy_max": float(A.max()), "copies": int(est.shape[0]), "runs_per_copy": int(est.shape[1]),
        "runs": int(est.size), "random": bool(raw["random"]),
        "estimate_mean": float(est.mean()) if est.max() > est.min() else float(est.flat[0]),
        "estimate_std": float(within.mean()),
        "estimate_between": float(per.std()) if per.max() > per.min() else 0.0,
        "estimate_min": float(est.min()), "estimate_max": float(est.max()),
        "mse": mse, "rms_error": math.sqrt(mse), "bias": float(err.mean()),
        "tracking": float(np.corrcoef(per, A)[0, 1]) if moves else None,
        "cost": float(ops.mean() / F), "cost_min": float(ops.min() / F), "cost_max": float(ops.max() / F),
        "ops_mean": float(ops.mean()), "draws_mean": float(raw["draws"].mean()), "forward_ops": int(forward_ops),
    }
