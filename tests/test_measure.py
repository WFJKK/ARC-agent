"""Measuring on copies of a network: how a copy is made, the seeds of a scoring's parts, what a scoring on copies
measures, and how its parts are put together."""

import math

import numpy as np
import pytest

from arcagent import lang, measure
from arcagent.networks import load_model

SPEC = {"count": 6, "runs": 5, "reference_inputs": 4096, "share": 0.125, "size": 0.04}


def world(model):
    return model.module().forward, model.task_module().answer, model.data_set, model.input_space


def test_a_copy_changes_a_share_of_the_weights_by_the_size_and_nothing_else(model):
    W = model.load_weights("rnn-b-s0")  # 32 weights in three matrices
    changed, steps = 0, []
    for j in range(2000):
        c = measure.copy_of(W, 0.125, 0.04, np.random.default_rng(j))
        assert sorted(c) == sorted(W)
        for k, w in W.items():
            assert c[k].dtype == np.float32 and c[k].shape == w.shape
            hit = c[k] != w
            changed += int(hit.sum())
            rms = math.sqrt(float((w.astype(np.float64) ** 2).mean()))
            steps.append((c[k].astype(np.float64) - w)[hit] / (0.04 * rms))
    assert abs(changed / (2000 * 32) - 0.125) < 0.005  # each weight on its own, with probability 0.125
    z = np.concatenate(steps)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03  # a normal draw of 0.04 times the matrix's RMS
    a, b = (measure.copy_of(W, 0.125, 0.04, np.random.default_rng(3)) for _ in range(2))
    assert all(np.array_equal(a[k], b[k]) for k in W)  # the generator decides the copy
    none = measure.copy_of(W, 1e-12, 0.04, np.random.default_rng(3))
    assert all(np.array_equal(none[k], W[k]) and none[k] is not W[k] for k in W)  # unchanged weights, bit for bit


def test_a_weight_that_is_zero_is_never_changed():
    m = load_model("models/lin-c")
    W = m.load_weights("lin-c-s0")
    assert (W["embed"][:, 64] == 0).all()  # the column of the slot that holds no token
    c = measure.copy_of(W, 1.0, 0.5, np.random.default_rng(0))
    assert (c["embed"][:, 64] == 0).all() and (c["embed"][:, :64] != W["embed"][:, :64]).all()
    assert all((c[k] != W[k]).all() for k in ("hidden", "unembed"))


def test_the_parts_of_a_scoring_get_seeds_of_their_own():
    seeds = [measure.derive(5, what, j) for what in ("copy", "reference", "runs") for j in range(200)]
    assert len(set(seeds)) == 600 and all(0 <= s < 2**64 for s in seeds)
    assert measure.derive(5, "copy", 3) == measure.derive(5, "copy", 3) != measure.derive(6, "copy", 3)


def test_the_copies_block_of_a_config_is_checked():
    assert measure.copies_spec(None) is None and measure.copies_spec({}) is None
    assert measure.copies_spec({**SPEC, "count": 0}) is None  # count 0: on the network itself
    assert measure.copies_spec({k: str(v) for k, v in SPEC.items()}) == SPEC
    with pytest.raises(ValueError, match="needs share, size"):
        measure.copies_spec({"count": 4, "runs": 2, "reference_inputs": 64, "share": None})
    with pytest.raises(ValueError, match="share is a probability"):
        measure.copies_spec({**SPEC, "share": 1.5})
    with pytest.raises(ValueError, match="size is above 0"):
        measure.copies_spec({**SPEC, "size": 0})


def test_a_scoring_on_copies_measures_each_copy_and_runs_the_estimator_on_it(model):
    W = model.load_weights("rnn-b-s0")
    fw, an, ds, space = world(model)
    e = model.monte_carlo(W, 16)
    raw = measure.on_copies(e, W, fw, an, ds, space, SPEC, seed=9)
    assert raw["accuracy"].shape == (6,) and raw["estimates"].shape == raw["ops"].shape == (6, 5) and raw["random"]
    assert raw["reference"] == {"method": "sample", "inputs": 4096, "chunk": 4096, "standard_error": raw["reference"]["standard_error"]}
    for j in range(6):
        w = measure.copy_of(W, 0.125, 0.04, np.random.default_rng(measure.derive(9, "copy", j)))
        assert raw["seeds"][j] == measure.derive(9, "copy", j)
        assert raw["accuracy"][j] == model.accuracy(w, 4096, measure.derive(9, "reference", j), 4096)["accuracy"]
        r = lang.run_estimator(e, w, 5, measure.derive(9, "runs", j))
        assert np.array_equal(raw["estimates"][j], r["estimates"]) and np.array_equal(raw["ops"][j], r["ops"])
    assert len(set(raw["accuracy"].tolist())) > 1  # the copies are not the network, and not each other
    F = model.forward_ops(W)
    p = measure.copies_parts(raw, F)
    err = raw["estimates"] - raw["accuracy"][:, None]
    assert math.isclose(p["mse"], float((err**2).mean())) and math.isclose(p["bias"], float(err.mean()))
    assert math.isclose(p["rms_error"], math.sqrt(p["mse"])) and math.isclose(p["cost"], 16 * 89 / 78)
    assert (p["copies"], p["runs_per_copy"], p["runs"]) == (6, 5, 30) and p["forward_ops"] == F
    assert math.isclose(p["accuracy"], raw["accuracy"].mean()) and math.isclose(p["accuracy_std"], raw["accuracy"].std())
    assert math.isclose(p["estimate_std"], raw["estimates"].std(axis=1).mean())
    assert math.isclose(p["estimate_between"], raw["estimates"].mean(axis=1).std())
    assert math.isclose(p["tracking"], np.corrcoef(raw["estimates"].mean(axis=1), raw["accuracy"])[0, 1])
    other = measure.on_copies(e, W, fw, an, ds, space, SPEC, seed=10)
    assert not np.array_equal(other["accuracy"], raw["accuracy"])  # another seed, other copies


def test_a_number_written_in_is_the_same_on_every_copy(model):
    W = model.load_weights("rnn-b-s0")
    g = model.graph(W)
    raw = measure.on_copies(g.entry(g.const(0.9)), W, *world(model), SPEC, seed=1)
    p = measure.copies_parts(raw, 78)
    assert (raw["estimates"] == 0.9).all() and not raw["random"] and (raw["ops"] == 0).all()
    assert p["estimate_mean"] == 0.9 and p["estimate_std"] == 0 and p["estimate_between"] == 0 and p["tracking"] is None
    assert math.isclose(p["mse"], float(((0.9 - raw["accuracy"]) ** 2).mean())) and p["cost"] == 0


def test_a_data_set_that_can_be_listed_is_measured_exactly_on_every_copy():
    m = load_model("models/lin-a")
    W = m.load_weights("lin-a-s0")
    spec = {"count": 3, "runs": 2, "reference_inputs": 65536, "share": 0.01, "size": 1.2}
    raw = measure.on_copies(m.monte_carlo(W, 8), W, *world(m), spec, seed=4)
    assert raw["reference"] == {"method": "exact", "inputs": 4096, "chunk": 32768, "standard_error": 0.0}
    x = np.indices([64, 64]).reshape(2, -1).T
    for j in range(3):
        w = measure.copy_of(W, 0.01, 1.2, np.random.default_rng(measure.derive(4, "copy", j)))
        out, ans = m.outputs(w, x)
        assert raw["accuracy"][j] == (out == ans).sum() / 4096


def test_an_estimator_that_fails_on_a_copy_is_refused_and_the_copy_is_named(model):
    """On the network itself this estimator gives 0; on a copy whose first input weight grew it takes the root of a
    negative number."""
    W = model.load_weights("rnn-b-s0")
    g = model.graph(W)
    e = g.entry(lang.sqrt(g.const(float(W["input"][0, 0])) - g.weight("input")[0, 0]))
    assert lang.run_estimator(e, W, 1, 0)["estimates"][0] == 0
    with pytest.raises(lang.EntryError, match=r"on a copy of the network \(copy\(\d+\) makes it, with the run seeds \[\d+, k\]\)"):
        measure.on_copies(e, W, *world(model), {**SPEC, "share": 1.0, "count": 12}, seed=2)
