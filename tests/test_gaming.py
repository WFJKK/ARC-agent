"""Ways to raise the reward without estimating better or computing less, and what stops each. A way found later gets
its test here before it is fixed. The last test is not a way around the reward but its known limit: it scores only
the estimate and its cost on one network, so a constant read off that network scores near the top; the held-out
networks are what show it."""

import numpy as np
import pytest

from arcagent import lang
from arcagent.networks import load_model
from arcagent.verifier import Verifier

CFG = {"verifier": {"runs": 256, "reference_inputs": 2**16}, "reward": {"name": "r_lambda", "lam": 1e-6}}


@pytest.fixture(scope="module")
def model():
    return load_model("models/rnn-b")


@pytest.fixture(scope="module")
def ver(model):
    return Verifier(model, "rnn-b-s0", CFG)


def test_work_done_once_from_the_weights_is_charged_to_every_run(model, ver):
    """Precomputing from the weights is no saving: every run pays for all the fixed work it reads."""
    g = model.graph(ver.weights)
    R = g.weight("recurrent")
    table = R
    for _ in range(20):
        table = table @ R  # 20 products of 4 x 4 matrices: 20 * 64 = 1280 operations, the same in every run
    x = g.normal(1).sum()
    e = g.entry(table.sum() * 0.0 + (x > 0.0).mean() * 0.0 + 0.98)
    r = ver.score(e, runs=64)
    assert r["parts"]["cost_min"] * ver.F >= 1280 and r["parts"]["random"]


def test_an_expensive_branch_is_charged_wherever_it_is_taken(model, ver):
    g = model.graph(ver.weights)
    coin = g.uniform(1).sum() < 0.5
    xs = g.inputs(200)
    expensive = (model.module().forward(g, lang, xs) == model.task_module().answer(g, lang, xs)).mean()
    e = g.entry(lang.cond(coin, expensive, g.const(0.98)))
    r = ver.score(e, runs=400, keep_runs=True)
    ops = np.array(r["runs_data"]["ops"])
    cheap, dear = ops.min(), ops.max()
    assert cheap < 10 and dear >= 200 * ver.sample_ops - 200  # the sampled branch pays for its 200 inputs when it runs
    assert 0.35 < (ops == dear).mean() < 0.65 and r["parts"]["cost"] == pytest.approx(ops.mean() / ver.F)


def test_a_table_of_answers_does_not_fit_in_the_constants(model, ver):
    g = model.graph(ver.weights)
    pasted = g.const(np.full(300, 0.98))
    r = ver.score(g.entry(pasted.mean()))
    assert r["status"] == "refused" and "constants hold more than the limit of 256" in r["reason"]


def test_the_verifier_draws_its_own_seed_so_no_estimator_can_be_fitted_to_its_draws(model, ver):
    e = model.monte_carlo(ver.weights, 32)
    a, b = ver.score(e, runs=64, keep_runs=True), ver.score(e, runs=64, keep_runs=True)
    # run by run the two scorings drew other inputs (their means alone tie about one time in twenty: 2,048 inputs
    # at 98.5% right)
    assert a["seed"] != b["seed"] and not np.array_equal(a["runs_data"]["estimates"], b["runs_data"]["estimates"])
    # and the reference inputs come from a seed no scoring uses: its own, fixed in the config, never shown
    assert ver.reference()["seed"] not in (a["seed"], b["seed"])


def test_one_draw_used_many_times_is_one_draw(model, ver):
    """Tiling one sampled input does not buy more samples: the run is charged one draw, and the estimate is as good
    as one sample (0 or 1)."""
    g = model.graph(ver.weights)
    one = g.inputs(1)
    tiled = lang.concat([one] * 50, axis=0)
    e = g.entry((model.module().forward(g, lang, tiled) == model.task_module().answer(g, lang, tiled)).mean())
    r = ver.score(e, runs=256)
    assert r["parts"]["draws_mean"] == 3 and set(np.unique([0.0, 1.0])) >= {r["parts"]["estimate_min"], r["parts"]["estimate_max"]}


def test_a_nan_in_some_runs_refuses_the_estimator(model, ver):
    g = model.graph(ver.weights)
    z = g.normal(1).sum()
    r = ver.score(g.entry(lang.where(z > 2.5, lang.log(g.const(0.0)), g.const(0.98))), runs=2000)
    assert r["status"] == "refused" and "not a finite number" in r["reason"]


def test_the_known_limit_a_constant_read_off_the_network(model, ver):
    """Not a way around the reward but what it cannot see: sampling outside the estimator is free, so the network's
    accuracy, measured in the sandbox and written in, scores near the top on the run's network. On another network
    the constant stays what it was, and sampling beats it there."""
    A = ver.reference()["accuracy"]
    g = model.graph(ver.weights)
    pasted = g.entry(g.const(round(A, 3)))
    sampling = model.monte_carlo(ver.weights, 128)
    p, s = ver.score(pasted), ver.score(sampling)
    assert p["reward"] > s["reward"] and p["reward"] > -1e-6
    other = Verifier(model, "rnn-b-s1", CFG)
    assert other.score(pasted)["reward"] < other.score(sampling)["reward"]
