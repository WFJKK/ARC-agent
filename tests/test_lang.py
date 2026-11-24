"""The estimator language: values, prices, random draws, laziness, what a run is charged, and what is refused."""

import json
import math

import numpy as np
import pytest

from arcagent import lang
from arcagent.lang import EntryError, Evaluator, Graph


def one_run(build, weights, seed=0):
    """Evaluate one expression of a new graph in one run: (value, operations, random numbers drawn)."""
    g = Graph(weights, {"kind": "normal"}, ["real"] * 3, seed=seed)
    e = build(g)
    r = Evaluator(g.entry(e), weights).run([seed])
    return r["estimate"], r["ops"], r["draws"]


def test_elementwise_prices_and_values(toy):
    A, v = toy["A"], toy["v"]
    val, ops, _ = one_run(lambda g: g.weight("A") * 2.0 + g.weight("v"), toy)
    assert np.allclose(val, A * 2.0 + v) and ops == 24  # 12 products, 12 sums
    val, ops, _ = one_run(lambda g: abs(-g.weight("v") - 1) // 2, toy)
    assert np.allclose(val, np.abs(-v - 1) // 2) and ops == 4 * 4  # neg, sub, abs, floordiv
    val, ops, _ = one_run(lambda g: (g.weight("v") < 0) & ~(g.weight("v") == 3.0), toy)
    assert (val == ((v < 0) & ~(v == 3.0))).all() and ops == 4 * 4  # lt, eq, not, and
    val, ops, _ = one_run(lambda g: g.weight("v") ** 2 + 2 ** g.weight("v"), toy)
    assert np.allclose(val, v**2 + 2**v) and ops == 12
    val, ops, _ = one_run(lambda g: lang.power(g.const(2), g.const(-1)), toy)
    assert val == 0.5 and ops == 1  # a whole number to a negative power is a fraction
    val, ops, _ = one_run(lambda g: lang.erf(g.weight("v")), toy)
    assert np.allclose(val, [math.erf(t) for t in v]) and ops == 4
    val, ops, _ = one_run(lambda g: lang.normal_cdf(g.weight("v")), toy)
    assert np.allclose(val, [0.5 * (1 + math.erf(t / math.sqrt(2))) for t in v]) and ops == 4 * 4  # mul, erf, add, mul


def test_matmul_counts_multiply_adds(toy):
    A, B, v = toy["A"], toy["B"], toy["v"]
    val, ops, _ = one_run(lambda g: g.weight("A") @ g.weight("B"), toy)
    assert np.allclose(val, A @ B) and ops == 3 * 4 * 5
    val, ops, _ = one_run(lambda g: g.weight("A") @ g.weight("v"), toy)
    assert np.allclose(val, A @ v) and ops == 3 * 4
    val, ops, _ = one_run(lambda g: g.weight("v") @ g.weight("v"), toy)
    assert np.isclose(val, v @ v) and ops == 4


def test_reductions_cost_n_minus_one(toy):
    A = toy["A"]
    for name, fn in (("sum", np.sum), ("max", np.max), ("min", np.min), ("argmax", np.argmax), ("argmin", np.argmin)):
        val, ops, _ = one_run(lambda g: getattr(g.weight("A"), name)(), toy)
        assert np.allclose(val, fn(A)) and ops == 11
        val, ops, _ = one_run(lambda g: getattr(g.weight("A"), name)(axis=1), toy)
        assert np.allclose(val, fn(A, axis=1)) and ops == 3 * 3
        val, ops, _ = one_run(lambda g: getattr(g.weight("A"), name)(axis=-2), toy)
        assert np.allclose(val, fn(A, axis=0)) and ops == 4 * 2
    val, ops, _ = one_run(lambda g: g.weight("A").mean(axis=1), toy)
    assert np.allclose(val, A.mean(axis=1)) and ops == 3 * 3 + 3  # a sum and a division


def test_along_an_axis(toy):
    A = toy["A"]
    val, ops, _ = one_run(lambda g: lang.cumsum(g.weight("A"), axis=1), toy)
    assert np.allclose(val, np.cumsum(A, axis=1)) and ops == 3 * 3
    val, ops, _ = one_run(lambda g: lang.sort(g.weight("A")), toy)
    assert np.allclose(val, np.sort(A, axis=1)) and ops == 3 * 4 * 2
    val, ops, _ = one_run(lambda g: lang.argsort(g.weight("A"), axis=0), toy)
    assert (val == np.argsort(A, axis=0, kind="stable")).all() and ops == 4 * 3 * 2
    val, ops, _ = one_run(lambda g: lang.softmax(g.weight("A")), toy)
    assert np.allclose(val.sum(axis=1), 1) and ops == 3 * 12


def test_reading_is_charged_per_entry_read(toy):
    T, K = toy["T"], toy["K"]
    val, ops, _ = one_run(lambda g: g.weight("T")[g.weight("K")[0, 1], g.weight("K")[2, 3]], toy)
    assert np.isclose(val, T[K[0, 1], K[2, 3]]) and ops == 1  # one number read; the static indices are free
    val, ops, _ = one_run(lambda g: g.weight("T")[:, g.weight("K")[0, 1]], toy)
    assert np.allclose(val, T[:, K[0, 1]]) and ops == 6  # one column
    val, ops, _ = one_run(lambda g: lang.take(g.weight("T"), g.weight("K")[0], axis=1), toy)
    assert np.allclose(val, T[:, K[0]]) and ops == 36
    val, ops, _ = one_run(lambda g: lang.take_sum(g.weight("T"), g.weight("K")[0, :2], axis=1), toy)
    assert np.allclose(val, T[:, K[0, 0]] + T[:, K[0, 1]]) and ops == 12


def test_moving_data_and_sources_are_free(toy):
    A = toy["A"]
    for build_fn, expect in (
        (lambda g: g.weight("A").T, A.T),
        (lambda g: g.weight("A").reshape(4, 3), A.reshape(4, 3)),
        (lambda g: g.weight("A")[1], A[1]),
        (lambda g: g.weight("A")[:, ::2], A[:, ::2]),
        (lambda g: g.weight("A")[None, :, 1], A[None, :, 1]),
        (lambda g: lang.concat([g.weight("A"), g.weight("A")], axis=1), np.concatenate([A, A], axis=1)),
        (lambda g: lang.stack([g.weight("v"), g.weight("v")]), np.stack([toy["v"], toy["v"]])),
        (lambda g: g.arange(5), np.arange(5)),
        (lambda g: g.const([[1.5, 2.0]]), np.array([[1.5, 2.0]])),
    ):
        val, ops, _ = one_run(build_fn, toy)
        assert val.shape == expect.shape and np.allclose(val, expect) and ops == 0


def test_random_draws_cost_one_per_number_and_are_seeded(toy):
    val, ops, draws = one_run(lambda g: g.normal([4, 3]), toy, seed=5)
    assert val.shape == (4, 3) and ops == draws == 12
    assert np.array_equal(val, np.random.default_rng([5, 0]).standard_normal((4, 3)))  # stream 0 of seed [5]
    val, ops, draws = one_run(lambda g: g.uniform(10), toy)
    assert ((0 <= val) & (val < 1)).all() and ops == draws == 10
    val, ops, draws = one_run(lambda g: g.integers(3, [100]), toy)
    assert set(np.unique(val)) <= {0, 1, 2} and val.dtype == np.int64 and ops == draws == 100
    val, ops, draws = one_run(lambda g: g.inputs(7), toy)
    assert val.shape == (7, 3) and ops == draws == 21


def test_two_random_nodes_are_independent_and_one_node_is_one_draw(toy):
    g = Graph(toy, seed=3)
    a, b = g.normal(1000), g.normal(1000)
    assert a.id != b.id and not np.allclose(a.value, b.value)
    same = (a - a).sum()
    ev = Evaluator(g.entry(same), toy)
    r = ev.run([3])
    assert r["estimate"] == 0 and r["draws"] == 1000  # one node is drawn once per run, however often it is used
    r1, r2 = ev.run([3, 0]), ev.run([3, 1])
    assert r1["ops"] == r2["ops"]


def test_the_builder_example_is_a_run_with_the_graph_seed(toy):
    g = Graph(toy, {"kind": "normal"}, ["real"] * 2, seed=11)
    x = g.inputs(50)
    est = (x[:, 0] > x[:, 1]).mean() + g.uniform(1).sum()
    entry = g.entry(est)
    assert np.isclose(Evaluator(entry, toy).run([11])["estimate"], est.value)
    assert not np.isclose(Evaluator(entry, toy).run([12])["estimate"], est.value)


def test_cond_charges_only_the_chosen_side(toy):
    g = Graph(toy)
    big = (g.weight("T") @ g.weight("T")).sum()  # 216 + 35
    small = g.weight("v").sum()  # 3
    e = g.entry(lang.cond(g.const(True), small, big))
    assert Evaluator(e, toy).run([0])["ops"] == 1 + 3
    g = Graph(toy)
    coin = g.uniform(1).sum() < 0.5  # a random condition: 1 draw, 0 for the sum, 1 for the comparison
    big = (g.weight("T") @ g.weight("T")).sum()
    small = g.weight("v").sum()
    e = g.entry(lang.cond(coin, small, big))
    ev = Evaluator(e, toy)
    seen = set()
    for k in range(40):
        r = ev.run([0, k])
        seen.add(r["ops"])
        assert r["ops"] in (2 + 1 + 3, 2 + 1 + 216 + 35)
    assert len(seen) == 2


def test_a_run_costs_what_it_would_cost_alone():
    """The fixed work kept between runs is charged to every run that uses it: a shared evaluator charges each run
    exactly what a fresh one does, whichever branches earlier runs took."""
    rng = np.random.default_rng(0)
    W = {"M": rng.normal(size=(5, 5)), "u": rng.normal(size=5)}
    for trial in range(30):
        g = Graph(W, seed=trial)
        r = np.random.default_rng(trial)
        pool = [g.weight("M"), g.weight("u")]
        fixed = [g.weight("M") @ g.weight("u"), lang.svd(g.weight("M"))[1], (g.weight("M") * 2.0).sum(axis=0)]
        pool += fixed
        coin = g.uniform(1).sum()
        for step in range(8):
            a = pool[int(r.integers(len(pool)))]
            b = pool[int(r.integers(len(pool)))]
            if a.shape != b.shape:
                b = a
            kind = int(r.integers(4))
            if kind == 0:
                pool.append(a + b)
            elif kind == 1:
                pool.append(lang.cond(coin < float(r.random()), a, b * 3.0))
            elif kind == 2:
                pool.append(lang.cond(g.const(bool(r.integers(2))), a.sum(), b.sum()))
            else:
                pool.append(a * g.normal(list(a.shape)) if step % 2 else lang.svd(g.weight("M") + coin)[0].sum() + a.sum())
        out = pool[-1].sum() + coin
        entry = g.entry(out)
        shared = Evaluator(entry, W)
        for k in range(12):
            alone = Evaluator(entry, W).run([trial, k])
            got = shared.run([trial, k])
            assert got["ops"] == alone["ops"] and got["draws"] == alone["draws"]
            assert np.allclose(got["estimate"], alone["estimate"])


def test_an_svd_is_charged_once_per_run(toy):
    g = Graph(toy)
    u, s, vt = lang.svd(g.weight("T"))
    e = g.entry((u @ vt).sum() + s.sum())
    ev = Evaluator(e, toy)
    expect = 4 * 6 * 6 * 6 + 6 * 6 * 6 + 35 + 5 + 1
    assert ev.run([0])["ops"] == expect and ev.run([1])["ops"] == expect  # the second run reads the kept values
    g = Graph(toy)
    s_only = lang.svd(g.weight("T"))[1].sum()
    assert Evaluator(g.entry(s_only), toy).run([0])["ops"] == 4 * 6 * 6 * 6 + 5


def test_run_estimator_runs_a_fixed_estimator_once(toy):
    g = Graph(toy)
    e = g.entry(g.weight("v").sum() / 10.0)
    r = lang.run_estimator(e, toy, runs=64, master=1)
    assert not r["random"] and len(r["estimates"]) == 64 and len(set(r["estimates"])) == 1
    assert set(r["ops"]) == {4} and set(r["draws"]) == {0}
    g = Graph(toy)
    e = g.entry((g.normal(16) > 0).mean())
    r = lang.run_estimator(e, toy, runs=200, master=1)
    assert r["random"] and len(set(r["estimates"])) > 3 and set(r["ops"]) == {16 + 16 + 15 + 1}
    again = lang.run_estimator(e, toy, runs=200, master=1)
    assert np.array_equal(r["estimates"], again["estimates"])  # a seed fixes every run
    assert not np.array_equal(r["estimates"], lang.run_estimator(e, toy, runs=200, master=2)["estimates"])
    g = Graph(toy)
    assert lang.uses_randomness(e) and not lang.uses_randomness(g.entry(g.const(0.5)))


def test_an_estimate_is_one_finite_number(toy):
    g = Graph(toy)
    with pytest.raises(EntryError, match="one number"):
        lang.run_estimator(g.entry(g.weight("v")), toy, 4, 0)
    g = Graph(toy)
    x = g.normal(1).sum()
    e = g.entry(lang.where(x > 1.5, g.const(0.0) / 0.0, x))  # nan on some runs only
    with pytest.raises(EntryError, match=r"run \d+ \(seed \[0, \d+\]\): the estimate is nan"):
        lang.run_estimator(e, toy, 400, 0)
    g = Graph(toy)
    r = lang.run_estimator(g.entry((g.weight("v") > 100.0).any()), toy, 3, 0)  # one boolean counts as 0 or 1
    assert list(r["estimates"]) == [0.0, 0.0, 0.0]


def test_refused_structures(toy):
    g = Graph(toy)
    good = g.entry(g.const(0.5))
    for bad, why in (
        ({**good, "predict": 0}, "an estimator is"),
        ({**good, "format": "level1-entry-v1"}, "an estimator is"),
        ({**good, "nodes": []}, "needs a list of nodes"),
        ({**good, "nodes": [{"op": "launch"}]}, "unknown operation"),
        ({**good, "nodes": [{"op": "const", "value": 1, "dtype": "int"}, {"op": "neg", "args": [1]}]}, "earlier nodes"),
        ({**good, "nodes": [{"op": "normal", "shape": [2], "stream": 0}, {"op": "normal", "shape": [2], "stream": 0}]},
         "used by another random node"),
        ({**good, "nodes": [{"op": "normal", "shape": [0], "stream": 0}]}, "positive whole numbers"),
        ({**good, "nodes": [{"op": "normal", "shape": [2]}]}, "stream must be"),
        ({**good, "nodes": [{"op": "const", "value": 1, "dtype": "int", "tokens": 1}]}, "does not take"),
        ({**good, "estimate": 3}, "estimate must name a node"),
    ):
        with pytest.raises(EntryError, match=why):
            lang.check_structure(bad)
    many = {"format": lang.FORMAT, "nodes": [{"op": "const", "value": [0.0] * 300, "dtype": "float"}], "estimate": 0}
    with pytest.raises(EntryError, match="constants hold more"):
        lang.check_structure(many, max_const_scalars=256)
    with pytest.raises(EntryError, match="above the limit of 1"):
        lang.check_structure({**good, "nodes": good["nodes"] * 2, "estimate": 0}, max_nodes=1)
    assert lang.check_structure(good) == {"nodes": 1, "const_scalars": 1, "random_nodes": 0}


def test_limits_and_time(toy):
    g = Graph(toy)
    with pytest.raises(EntryError, match="above the limit"):
        g.normal([3000, 3000])
    chain = [{"op": "add", "args": [0, 1]}] + [{"op": "add", "args": [40 + k, 2 + k]} for k in range(38)]
    big = {"format": lang.FORMAT, "nodes": [{"op": "normal", "shape": [2000, 1000], "stream": s} for s in range(40)] + chain,
           "estimate": 78}
    with pytest.raises(EntryError, match="holds more than"):
        Evaluator(big, toy).run([0])
    slow = {"format": lang.FORMAT, "nodes": [{"op": "normal", "shape": [10], "stream": 0}], "estimate": 0}
    with pytest.raises(EntryError, match="ran out of time"):
        Evaluator(slow, toy, deadline=0.0).run([0])


def test_load_keeps_streams_and_adds_new_ones(toy):
    g = Graph(toy, seed=4)
    a = g.normal(3)
    b = g.normal(3)
    entry = g.entry((a * b).sum())
    g2, est = Graph.load(entry, toy, seed=4)
    c = g2.normal(3)
    entry2 = g2.entry(est + c.sum())
    lang.check_structure(entry2)
    assert sorted(n["stream"] for n in entry2["nodes"] if n["op"] == "normal") == [0, 1, 2]
    assert np.isclose(Evaluator(entry, toy).run([4])["estimate"], est.value)
    assert json.loads(lang.canonical(entry)) == entry and len(lang.entry_hash(entry)) == 64
    assert "<- estimate" in lang.show(entry)
