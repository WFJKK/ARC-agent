"""The verifier: the reference accuracy, the runs, the parts, the receipt, and what it refuses."""

import math

import numpy as np
import pytest

from arcagent import lang
from arcagent.networks import load_model
from arcagent.verifier import Verifier, agent_view, describe

CFG = {"verifier": {"reference_inputs": 2**17, "runs": 512}, "reward": {"name": "r_lambda", "lam": 1e-6}}


@pytest.fixture(scope="module")
def model():
    return load_model("models/rnn-b")


@pytest.fixture(scope="module")
def ver(model):
    return Verifier(model, "rnn-b-s0", CFG)


def const(model, ver, c):
    g = model.graph(ver.weights)
    return g.entry(g.const(float(c)))


def test_the_reference_is_measured_once_and_can_be_handed_in(model, ver):
    ref = ver.reference()
    assert ref["n"] == 2**17 and ref["network"] == "rnn-b-s0" and 0.97 < ref["accuracy"] < 0.995
    assert ver.reference() is ref
    other = Verifier(model, "rnn-b-s0", CFG, reference=ref)
    assert other.reference() == ref
    with pytest.raises(ValueError, match="not the one the config asks for"):
        Verifier(model, "rnn-b-s0", {**CFG, "verifier": {"reference_inputs": 2**16}}, reference=ref)
    with pytest.raises(ValueError, match="not the one the config asks for"):
        Verifier(model, "rnn-b-s1", CFG, reference=ref)


def test_a_constant_is_one_run_repeated(model, ver):
    A = ver.reference()["accuracy"]
    r = ver.score(const(model, ver, 0.9))
    p = r["parts"]
    assert r["status"] == "scored" and r["runs"] == 512 and p["runs"] == 512 and not p["random"]
    assert p["estimate_std"] == 0 and p["cost"] == 0 and p["draws_mean"] == 0
    assert math.isclose(p["mse"], (0.9 - A) ** 2) and math.isclose(p["bias"], 0.9 - A)
    assert math.isclose(r["reward"], -((0.9 - A) ** 2))
    assert r["benchmark"]["mse_reduction"] is None  # no sampling at no cost


def test_plain_sampling_costs_what_it_draws_and_errs_as_it_should(model, ver):
    A = ver.reference()["accuracy"]
    n = 64
    r = ver.score(model.monte_carlo(ver.weights, n), seed=7, runs=4096)
    p, b = r["parts"], r["benchmark"]
    assert p["random"] and p["draws_mean"] == n * model.n_ctx
    assert math.isclose(p["cost"], n * ver.sample_ops / ver.F)  # n inputs at S operations each, nothing else
    assert p["cost_min"] == p["cost_max"] == p["cost"]
    assert abs(p["mse"] / (A * (1 - A) / n) - 1) < 0.1  # Var / n, within the noise of 4096 runs
    assert abs(p["bias"]) < 4 * math.sqrt(A * (1 - A) / n / 4096)
    assert abs(b["mse_reduction"]) < 0.1  # sampling is its own benchmark
    assert math.isclose(r["reward"], -p["mse"] - 1e-6 * p["cost"])
    again = ver.score(model.monte_carlo(ver.weights, n), seed=7, runs=4096)
    assert again["parts"] == p  # a seed fixes every run
    other = ver.score(model.monte_carlo(ver.weights, n), runs=4096)
    assert other["seed"] != 7 and other["parts"]["estimate_mean"] != p["estimate_mean"]  # a fresh seed by default


def test_the_runs_can_be_kept(model, ver):
    r = ver.score(model.monte_carlo(ver.weights, 8), runs=16, keep_runs=True)
    d = r["runs_data"]
    assert len(d["estimates"]) == len(d["ops"]) == len(d["draws"]) == 16
    assert math.isclose(np.mean(d["estimates"]), r["parts"]["estimate_mean"])


def test_what_the_agent_is_shown(model, ver):
    r = ver.score(model.monte_carlo(ver.weights, 8), runs=64)
    v = agent_view(r)
    assert set(v) == {"status", "reason", "reward", "size", "runs", "mean_squared_error", "rms_error", "cost", "cost_min",
                      "cost_max", "estimate_mean", "estimate_std", "random_draws_per_run", "uses_random_draws"}
    text = str(v)
    assert str(round(r["parts"]["accuracy"], 6)) not in text and "bias" not in text and "benchmark" not in text
    assert "rms error" in describe(r) and "sampling's at that cost" in describe(r)


def test_refusals_say_why(model, ver):
    assert ver.score({"format": "level1-entry-v1"})["reason"].startswith("an estimator is")
    g = model.graph(ver.weights)
    x = g.normal(1).sum()
    nan_sometimes = g.entry(lang.where(x > 2.0, g.const(0.0) / 0.0, x))
    r = ver.score(nan_sometimes, seed=3, runs=2000)
    assert r["status"] == "refused" and "the estimate is nan" in r["reason"] and "seed [3," in r["reason"]
    g = model.graph(ver.weights)
    r = ver.score(g.entry(g.inputs(4)))
    assert r["status"] == "refused" and "one number" in r["reason"]
    big = Verifier(model, "rnn-b-s0", {**CFG, "entry": {"max_nodes": 5}})
    r = big.score(model.monte_carlo(ver.weights, 4))
    assert r["status"] == "refused" and "above the limit of 5" in r["reason"]
    slow = Verifier(model, "rnn-b-s0", {**CFG, "verifier": {**CFG["verifier"], "time_limit_s": 0.0}})
    r = slow.score(model.monte_carlo(ver.weights, 4))
    assert r["status"] == "refused" and "ran out of time" in r["reason"]
    assert describe(r).startswith("refused:")


# scoring on copies of the network ----------------------------------------------------------------
COPIES = {"count": 8, "runs": 16, "reference_inputs": 4096, "share": 0.125, "size": 0.04}
CFG_COPIES = {**CFG, "verifier": {**CFG["verifier"], "copies": COPIES}}


@pytest.fixture(scope="module")
def cver(model, ver):
    return Verifier(model, "rnn-b-s0", CFG_COPIES, reference=ver.reference())


def test_a_scoring_on_copies_is_what_the_measuring_module_measures(model, ver, cver):
    import json

    from arcagent import measure

    e = model.monte_carlo(cver.weights, 64)
    r = cver.score(e, seed=3, keep_runs=True)
    assert r["status"] == "scored" and r["scored_on"] == "copies" and r["runs"] == 128 and r["seed"] == 3
    raw = measure.on_copies(e, cver.weights, model.module().forward, model.task_module().answer, model.data_set,
                            model.input_space, cver.copies, 3, chunk=cver.chunk())
    p = r["parts"]
    assert p == measure.copies_parts(raw, cver.F) and math.isclose(r["reward"], -p["mse"] - 1e-6 * p["cost"])
    A, b = raw["accuracy"], r["benchmark"]
    assert r["copies"]["accuracy"] == A.tolist() and r["copies"]["seeds"] == raw["seeds"] and r["copies"]["count"] == 8
    assert (r["copies"]["share"], r["copies"]["size"], r["copies"]["runs"]) == (0.125, 0.04, 16)
    assert r["reference"] == {"method": "sample", "inputs": 4096, "chunk": 4096, "copies": 8,
                              "standard_error": raw["reference"]["standard_error"]}
    # the benchmark: plain sampling on these copies, and what a number written in could score on them
    s, v = cver.sample_ops / cver.F, float(np.mean(A * (1 - A)))
    assert math.isclose(b["variance"], v) and math.isclose(b["samples_at_cost"], 64)
    assert math.isclose(b["mse_at_cost"], v / 64) and math.isclose(b["best_reward"], -2 * math.sqrt(1e-6 * s * v))
    assert math.isclose(b["constant_best"], -A.var())
    assert math.isclose(b["constant_unchanged"], -float(np.mean((ver.reference()["accuracy"] - A) ** 2)))
    kept = r["runs_data"]
    assert np.array(kept["estimates"]).shape == (8, 16) and kept["accuracy"] == A.tolist()
    assert np.array_equal(np.array(kept["estimates"]), raw["estimates"])
    # what the agent is told: nothing of the copies' accuracies
    seen = agent_view(r)
    assert set(seen) == {"status", "reason", "reward", "size", "runs", "mean_squared_error", "rms_error", "cost", "cost_min",
                         "cost_max", "estimate_mean", "estimate_std", "random_draws_per_run", "uses_random_draws", "copies",
                         "runs_per_copy", "estimate_std_between_copies"}
    assert "accuracy" not in json.dumps(seen) and seen["copies"] == 8 and seen["runs_per_copy"] == 16
    assert "over 128 runs on 8 copies" in describe(r)
    assert cver.score(e, seed=3)["parts"] == p  # the seed decides the copies and the runs


def test_scoring_on_the_network_itself_stays_what_it_was(model, ver, cver):
    e = model.monte_carlo(cver.weights, 64)
    alone = cver.score(e, seed=3, copies=False)
    assert alone["scored_on"] == "network" and alone["runs"] == 512 and "copies" not in alone and "copies" not in alone["parts"]
    plain = ver.score(e, seed=3)
    assert plain["parts"] == alone["parts"] and plain["reward"] == alone["reward"] and plain["scored_on"] == "network"
    assert "constant_best" not in plain["benchmark"]
    with pytest.raises(ValueError, match="no copies block"):
        ver.score(e, copies=True)
    with pytest.raises(ValueError, match="needs share, size"):
        Verifier(model, "rnn-b-s0", {**CFG, "verifier": {"copies": {"count": 4, "runs": 2, "reference_inputs": 64}}})


def test_on_copies_an_estimator_that_draws_nothing_gives_one_number_for_each_copy(model, cver):
    g = model.graph(cver.weights)
    e = g.entry(g.weight("readout").sum() + g.weight("recurrent").sum() + g.weight("input").sum())  # follows the copy, draws nothing
    r = cver.score(e, seed=5, keep_runs=True)
    est = np.array(r["runs_data"]["estimates"])
    assert not r["parts"]["random"] and (est == est[:, :1]).all() and r["parts"]["estimate_std"] == 0
    assert len(set(est[:, 0].tolist())) > 4 and r["parts"]["estimate_between"] > 0  # the copies' weights differ


def test_on_copies_a_refusal_names_the_copy(model, cver):
    g = model.graph(cver.weights)
    e = g.entry(lang.sqrt(g.const(float(cver.weights["input"][0, 0])) - g.weight("input")[0, 0]))
    assert cver.score(e, copies=False)["status"] == "scored"  # on the network itself it is fine
    r = Verifier(model, "rnn-b-s0", {**CFG, "verifier": {**CFG["verifier"], "copies": {**COPIES, "share": 1.0, "count": 12}}},
                 reference=cver.reference()).score(e, seed=2)
    assert r["status"] == "refused" and r["reason"].startswith("on a copy of the network (copy(") and r["parts"] is None
    assert "the estimate is" in r["reason"]
