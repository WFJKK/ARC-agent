"""The withheld reference estimators that compute the accuracy from the weights: each gives the network's accuracy
on the network itself and on its copies. They are what a scoring on copies is meant to leave room for, an estimate
that follows the weights, and they say how much room there is: on the linear network going through every input
costs about what the best plain sampling loses, and four more steps bring the loss to a little over a third of
it; on the attention network every input costs well under sampling's loss; and the two-unit recurrent network has
an exact estimator at about a sixth of it."""

import math

import numpy as np
import pytest

from arcagent import lang, measure, run_config
from arcagent.networks import load_model
from arcagent.verifier import Verifier

SMALL = {"count": 12, "runs": 4}


def setup(name, **copies):
    cfg = run_config.load(f"config/pilot/{name}.yaml", {"runs_dir": "/nonexistent", "verifier": {"copies": {**SMALL, **copies}}})
    m = load_model(cfg["model"])
    return m, Verifier(m, cfg["network"], cfg)


# model, estimator, and its loss as a share of the best plain sampling's on the same copies: at most this
@pytest.mark.parametrize("name,estimator,share", [("lin-a", "the table of token logits, every input", 1.15),
                                                  ("att-a", "tables over single tokens, every input", 0.7)])
def test_an_estimator_that_goes_through_every_input_is_exact_on_the_network_and_on_every_copy(name, estimator, share):
    m, v = setup(name)
    e = m.reference_entries(v.weights)[estimator]
    for net in m.network_ids():
        W = m.load_weights(net)
        assert lang.run_estimator(e, W, 1, 0)["estimates"][0] == pytest.approx(m.accuracy(W, 2**21, 0)["accuracy"], abs=1e-12)
    r = v.score(e, seed=2)
    p = r["parts"]
    assert r["status"] == "scored" and not p["random"] and p["rms_error"] < 1e-12 and p["tracking"] == pytest.approx(1.0)
    assert r["reference"]["method"] == "exact" and p["accuracy_std"] > 0.02  # the copies do differ
    assert r["reward"] == pytest.approx(-1e-6 * p["cost"]) and r["reward"] >= share * r["benchmark"]["best_reward"]
    assert r["reward"] > 5 * r["benchmark"]["constant_best"]  # far above any number written in


def test_sampling_from_the_table_of_token_logits_is_cheaper_than_sampling_the_network():
    m, v = setup("lin-a", count=48, runs=16)
    e = m.reference_entries(v.weights)["the table of token logits, 448 sampled inputs"]
    r = v.score(e, seed=2)
    p, b = r["parts"], r["benchmark"]
    assert p["random"] and p["cost"] < 200 and b["samples_at_cost"] < 200  # 448 inputs at the price of 189 sampled ones
    assert abs(p["bias"]) < 0.004 and p["mse"] < 0.6 * b["mse_at_cost"]
    assert r["reward"] > 0.8 * b["best_reward"]  # at least a fifth less loss than the best plain sampling


def test_the_linear_networks_estimators_in_the_order_they_were_found():
    """Each step on lin-a against the best plain sampling on the same copies: sampling from the table, sampling
    whole permutations from it, counting over every pair with only the rows that can win, and that count with one
    fitted factor. The count is never below a copy's accuracy."""
    m, v = setup("lin-a", count=64, runs=16)
    ests = m.reference_entries(v.weights)
    got = {name: v.score(e, seed=3, keep_runs=True) for name, e in ests.items()}
    best = next(iter(got.values()))["benchmark"]["best_reward"]  # the same copies for all: the seed makes them
    loss = {name: r["reward"] / best for name, r in got.items()}  # 1: what the best plain sampling loses
    assert 0.55 < loss["the table of token logits, 448 sampled inputs"] < 0.75
    assert 0.5 < loss["the table of token logits, 6 permutations"] < 0.68
    assert 0.38 < loss["the top 4 rows of each token's column, every input"] < 0.48
    assert 0.3 < loss["the top 3 rows of each token's column, every input, error rate times 1.098"] < 0.42
    top4 = got["the top 4 rows of each token's column, every input"]
    over = np.array(top4["copies"]["estimate_mean"]) - np.array(top4["copies"]["accuracy"])
    assert not top4["parts"]["random"] and over.min() >= 0 and 0.002 < over.mean() < 0.008 and top4["parts"]["cost"] < 185
    perm = got["the table of token logits, 6 permutations"]
    assert abs(perm["parts"]["bias"]) < 0.004 and perm["parts"]["draws_mean"] == 6 * 64
    # on the network itself the two counts are the accuracy, or all but
    for name, least in (("the top 4 rows of each token's column, every input", 4095 / 4096),
                        ("the top 3 rows of each token's column, every input, error rate times 1.098", 1 - 1.098 / 4096)):
        assert lang.run_estimator(ests[name], v.weights, 1, 0)["estimates"][0] == pytest.approx(least, abs=1e-12)


def test_the_region_estimator_of_the_two_unit_network_is_the_accuracy():
    """Checked against the network run on a fine grid of directions: on the model's five networks (two of which
    give one output on every input) and on copies, some of which have stretches where no unit is on."""
    m, v = setup("rnn-a", reference_inputs=2**18)
    e = m.reference_entries(v.weights)["the regions on which the logits are linear, exactly"]

    def on_a_grid(w, n=200000):
        th = (np.arange(n) + 0.5) / n * 2 * np.pi
        x = np.stack([np.cos(th), np.sin(th)], axis=1)
        out, ans = m.outputs(w, x)
        return float((out == ans).mean())

    for net in m.network_ids():
        W = m.load_weights(net)
        assert lang.run_estimator(e, W, 1, 0)["estimates"][0] == pytest.approx(on_a_grid(W), abs=5e-5)
    spec = v.copies
    for j in (172, 201, 219, 3, 4, 5):  # the first three have a stretch where no unit is on
        w = measure.copy_of(v.weights, spec["share"], spec["size"], np.random.default_rng(j))
        assert lang.run_estimator(e, w, 1, 0)["estimates"][0] == pytest.approx(on_a_grid(w), abs=5e-5)
    r = v.score(e, seed=2)
    p = r["parts"]
    assert not p["random"] and p["cost"] < 80 and p["rms_error"] < 0.002  # the copies' accuracies are sampled: 2^18 inputs
    assert r["reward"] > r["benchmark"]["best_reward"] / 4 and math.isclose(p["cost"], 1434 / 19)
