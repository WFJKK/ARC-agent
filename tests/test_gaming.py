"""Ways to raise the reward without estimating better or computing less, and what stops each. A way found later gets
its test here before it is fixed.

The largest of them is not a trick inside an estimator but what a reward on one network cannot see: the agent's own
code may run the network as often as it likes at no charge, so the accuracy, measured there and written into an
estimator as a constant, scores near the top (test_the_known_limit_...). Scoring on copies of the network closes
it (the tests after that one): what is computed from the weights follows a copy, a number written in does not."""

import math

import numpy as np
import pytest

from arcagent import lang, measure
from arcagent.networks import load_model
from arcagent.verifier import Verifier

CFG = {"verifier": {"runs": 256, "reference_inputs": 2**16}, "reward": {"name": "r_lambda", "lam": 1e-6}}
# rnn-b-s0's copies as its pilot makes them (config/pilot/rnn-b.yaml), fewer of them and measured on fewer inputs
COPIES = {"count": 64, "runs": 16, "reference_inputs": 8192, "share": 0.125, "size": 0.04}
CFG_COPIES = {**CFG, "verifier": {**CFG["verifier"], "copies": COPIES}}


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
    """Scored on the network itself, this is what the reward cannot see: sampling outside the estimator is free, so
    the network's accuracy, measured in the sandbox and written in, scores near the top on the run's network. On
    another network the constant stays what it was, and sampling beats it there."""
    A = ver.reference()["accuracy"]
    g = model.graph(ver.weights)
    pasted = g.entry(g.const(round(A, 3)))
    sampling = model.monte_carlo(ver.weights, 128)
    p, s = ver.score(pasted), ver.score(sampling)
    assert p["reward"] > s["reward"] and p["reward"] > -1e-6
    other = Verifier(model, "rnn-b-s1", CFG)
    assert other.score(pasted)["reward"] < other.score(sampling)["reward"]


# scoring on copies of the network ----------------------------------------------------------------
@pytest.fixture(scope="module")
def cver(model, ver):
    return Verifier(model, "rnn-b-s0", CFG_COPIES, reference=ver.reference())


def best_sampling(model, v, seed):
    """Plain sampling at the number of inputs that is best under the reward on a scoring's copies, scored on them."""
    probe = v.score(model.monte_carlo(v.weights, 8), seed=seed)
    return v.score(model.monte_carlo(v.weights, round(probe["benchmark"]["best_samples"])), seed=seed)


def test_on_copies_a_number_written_in_loses_to_plain_sampling(model, ver, cver):
    """The known limit, closed. The accuracy read off the network is all but perfect on the network itself and far
    off on its copies; and no number does better there than the copies' own mean accuracy, which nobody is told and
    which loses to plain sampling too."""
    A = ver.reference()["accuracy"]
    g = model.graph(cver.weights)
    pasted = g.entry(g.const(round(A, 3)))
    assert cver.score(pasted, copies=False)["reward"] > -1e-6
    for seed in (13, 14):
        p, s = cver.score(pasted, seed=seed), best_sampling(model, cver, seed)
        assert p["reward"] < 5 * s["reward"] < 0  # at least five times the loss
        assert math.isclose(p["reward"], p["benchmark"]["constant_unchanged"], rel_tol=0.02)  # A, rounded to three places
        g = model.graph(cver.weights)
        best = cver.score(g.entry(g.const(p["parts"]["accuracy"])), seed=seed)  # the best number there is, in hindsight
        assert math.isclose(best["reward"], p["benchmark"]["constant_best"]) and best["reward"] < 2 * s["reward"]
        assert p["parts"]["tracking"] is None and s["parts"]["tracking"] > 0.95  # sampling follows the copies


def test_every_scoring_makes_its_own_copies(model, cver):
    """Nothing a scoring tells about its copies is of use in the next: the copies are made from the scoring's seed,
    which the verifier draws afresh. With the same seed a scoring can be made again, so a receipt can be checked."""
    e = model.monte_carlo(cver.weights, 32)
    a, b = cver.score(e, count=8), cver.score(e, count=8)
    assert a["seed"] != b["seed"] and not set(a["copies"]["seeds"]) & set(b["copies"]["seeds"])
    assert a["copies"]["accuracy"] != b["copies"]["accuracy"]
    again = cver.score(e, seed=a["seed"], count=8)
    assert again["copies"] == a["copies"] and again["parts"] == a["parts"]
    # the network itself is none of the copies: every one of them has weights changed
    for seed in a["copies"]["seeds"]:
        w = measure.copy_of(cver.weights, COPIES["share"], COPIES["size"], np.random.default_rng(seed))
        assert any((w[k] != cver.weights[k]).any() for k in w)


def test_a_table_of_what_each_weight_does_loses_to_plain_sampling(model, ver, cver):
    """The next thing brute force can try: carry the network's weights into the estimator as constants, look how a
    copy differs from them, and predict the copy's accuracy from a table made beforehand of how the accuracy moves
    with each weight (here a quadratic per weight, fitted by running the network with that weight alone changed:
    97 numbers for the 32 weights, well within the limit). It follows the copies in part, since the changes of
    several weights do not simply add up, and plain sampling beats it."""
    W0 = cver.weights
    fw, an = model.module().forward, model.task_module().answer
    sampler = measure.monte_carlo(fw, an, W0, model.data_set, model.input_space, 8192)

    def acc(w):
        return measure.accuracy(fw, an, w, model.data_set, model.input_space, 4 * 8192, 3, 8192, sampler)["accuracy"]

    A0 = acc(W0)
    g = model.graph(W0)
    est = g.const(A0)
    for k in sorted(W0):
        w = W0[k].astype(np.float64)
        steps = np.array([-2.0, -1.0, 1.0, 2.0]) * COPIES["size"] * math.sqrt(float((w**2).mean()))
        a, b = np.zeros_like(w), np.zeros_like(w)
        for idx in np.ndindex(w.shape):
            moved = []
            for st in steps:
                c = {n: x.copy() for n, x in W0.items()}
                c[k][idx] = w[idx] + st
                moved.append(acc(c) - A0)
            a[idx], b[idx] = np.linalg.lstsq(np.stack([steps, steps**2], axis=1), np.array(moved), rcond=None)[0]
        d = g.weight(k) - g.const(w)
        est = est + (d * g.const(a)).sum() + (d * d * g.const(b)).sum()
    table = g.entry(est)
    for seed in (13, 14):
        t, s = cver.score(table, seed=seed), best_sampling(model, cver, seed)
        assert t["status"] == "scored" and t["size"]["const_scalars"] == 97 and t["parts"]["cost"] < 5
        assert 0.7 < t["parts"]["tracking"] < 0.97  # it is not nothing
        assert t["reward"] < 1.5 * s["reward"] < 0  # and it loses: at least one and a half times sampling's loss
        assert t["reward"] > t["benchmark"]["constant_best"]  # though it beats every number written in
