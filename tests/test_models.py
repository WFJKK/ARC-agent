"""The models: the network and the task in the language agree with plain numpy, a forward pass is priced as in
level 1, the verifier's accuracy measurement counts what numpy counts, and the weights are AlgZoo's files."""

import importlib.util
import math

import numpy as np
import pytest

from arcagent import lang
from arcagent.networks import ModelError, load_model
from arcagent.util import REPO_ROOT, read_yaml


def numpy_forward(W, x):
    w, R, U = (W[k].astype(np.float64) for k in ("input", "recurrent", "readout"))
    h = np.maximum(x[:, :1] * w[None, :, 0], 0)
    for t in range(1, x.shape[1]):
        h = np.maximum(x[:, t : t + 1] * w[None, :, 0] + h @ R.T, 0)
    return (h @ U.T).argmax(axis=1)


def numpy_answer(x):
    return np.argsort(x, axis=1, kind="stable")[:, -2]


@pytest.fixture(scope="module")
def rnn_g():
    return load_model("models/rnn-g")


@pytest.fixture(scope="module")
def rnn_b():
    return load_model("models/rnn-b")


@pytest.mark.parametrize("name,F,A,S", [("rnn-g", 2937, 40, 2989), ("rnn-b", 78, 6, 89)])
def test_prices_of_a_forward_pass_the_answer_and_a_sample(name, F, A, S):
    m = load_model(f"models/{name}")
    W = m.load_weights(m.network_ids()[0])
    assert m.forward_ops(W) == F  # level 1's count for the same network
    assert m.answer_ops(W) == A
    assert m.sample_ops(W) == S  # draws (n_ctx) + forward pass + answer + one comparison + one more term of the sum
    assert S == m.n_ctx + F + A + 2


@pytest.mark.parametrize("name", ["rnn-g", "rnn-b"])
def test_the_network_and_the_task_agree_with_numpy(name):
    m = load_model(f"models/{name}")
    x = np.random.default_rng(1).standard_normal((3000, m.n_ctx))
    for nid in m.network_ids():
        W = m.load_weights(nid)
        g = m.graph(W)
        xs = g.const(x)
        out = lang.Evaluator(g.entry(m.module().forward(g, lang, xs)), W).run([0])["estimate"]
        assert np.array_equal(out, numpy_forward(W, x)), nid
    g = m.graph(W)
    ans = lang.Evaluator(g.entry(m.task_module().answer(g, lang, g.const(x))), W).run([0])["estimate"]
    assert np.array_equal(ans, numpy_answer(x))


def test_the_accuracy_measurement_counts_what_numpy_counts(rnn_b, rnn_g):
    for m in (rnn_b, rnn_g):
        nid = m.network_ids()[0]
        W = m.load_weights(nid)
        got = m.accuracy(W, n=4096, seed=5, chunk=1024)
        # chunk k is monte_carlo(1024) run with seed [5, k]; its inputs are the first random node, stream 0
        x = np.concatenate([np.random.default_rng([5, k, 0]).standard_normal((1024, m.n_ctx)) for k in range(4)])
        assert got["correct"] == int((numpy_forward(W, x) == numpy_answer(x)).sum())
        assert got["accuracy"] == got["correct"] / 4096 and math.isclose(
            got["standard_error"], math.sqrt(got["accuracy"] * (1 - got["accuracy"]) / 4096))
    with pytest.raises(ValueError, match="multiple of the chunk"):
        rnn_b.accuracy(rnn_b.load_weights("rnn-b-s0"), n=1000, seed=0, chunk=300)


def test_the_recorded_accuracies_are_the_verifiers_own(rnn_b):
    """task.yaml records the verifier's measurement on 2^21 inputs, seed 20261124; checked here on the small model
    (rnn-g's are checked by the slow test below)."""
    rec = read_yaml(REPO_ROOT / "withheld" / "rnn-b" / "task.yaml")["accuracy"]
    for nid in rnn_b.network_ids():
        got = rnn_b.accuracy(rnn_b.load_weights(nid), n=2**21, seed=20261124)
        assert round(got["accuracy"], 6) == rec[nid]


@pytest.mark.slow
def test_the_recorded_accuracies_of_rnn_g(rnn_g):
    rec = read_yaml(REPO_ROOT / "withheld" / "rnn-g" / "task.yaml")["accuracy"]
    for nid in rnn_g.network_ids():
        got = rnn_g.accuracy(rnn_g.load_weights(nid), n=2**21, seed=20261124)
        assert round(got["accuracy"], 6) == rec[nid]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_weights_are_algzoos_files_converted(rnn_b, rnn_g):
    conv = _load(REPO_ROOT / "withheld" / "rnn" / "convert.py", "convert")
    for m in (rnn_b, rnn_g):
        for nid, _file in conv.NETWORKS[m.name]:
            W = m.load_weights(nid)
            w = conv.convert(m.name, nid)
            assert set(w) == set(W) and all(np.array_equal(w[k], W[k]) and W[k].dtype == np.float32 for k in w)


def test_the_noised_networks_are_what_the_script_makes(rnn_g):
    noise = _load(REPO_ROOT / "withheld" / "rnn-g" / "noise.py", "noise")
    first = rnn_g.load_weights("rnn-g-s0")
    for k in noise.SIGMAS:
        W = rnn_g.load_weights(f"rnn-g-n{k}")
        w = noise.noised(first, k)
        assert all(np.array_equal(w[n], W[n]) for n in W)


def test_a_model_folder_is_checked(rnn_b, tmp_path):
    with pytest.raises(ModelError, match="not a network"):
        rnn_b.network("rnn-b-s9")
    bad = tmp_path / "m"
    bad.mkdir()
    (bad / "config.yaml").write_text((rnn_b.path / "config.yaml").read_text().replace("  says: >", "  said: >"))
    with pytest.raises(ModelError, match="sentences the agent is told"):
        load_model(bad)
    info = rnn_b.info()
    assert set(info) == {"sizes", "input_space", "n_outputs", "weights", "data_set", "max_batch"}
    assert "says" not in info["data_set"] and info["max_batch"] == 32768
