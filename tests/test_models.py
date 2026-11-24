"""The models: the networks and the tasks in the language agree with plain numpy, a forward pass is priced as in
level 1, the verifier's accuracy measurement counts what numpy counts, exactly where the data set can be listed,
and the weights are the files they were made from."""

import importlib.util
import math
import shutil

import numpy as np
import pytest

from arcagent import lang
from arcagent.networks import ModelError, load_model
from arcagent.util import REPO_ROOT, read_yaml

MODELS = sorted(p.parent.name for p in (REPO_ROOT / "models").glob("*/config.yaml"))

# model: (F, the answer's operations per input, plain sampling's price of one more input). F is level 1's count for
# the same network; level 1 prices every operation as this repository does.
PRICES = {
    "rnn-g": (2937, 40, 2989), "rnn-b": (78, 6, 89), "rnn-a": (19, 2, 25), "rnn-d": (19, 2, 25),
    "rnn-e": (78, 6, 89), "rnn-f": (2937, 40, 2989),
    "lin-a": (655, 1, 660), "lin-a-1to4": (671, 11, 698), "lin-b": (703, 23, 754), "lin-b-4": (671, 11, 688),
    "lin-c": (671, 3, 690), "lin-d": (703, 7, 738), "lin-e": (671, 3, 690), "lin-f": (703, 7, 738),
    "lin-f-4": (671, 11, 688),
    "att-a": (8621, 5, 8636), "att-b": (8621, 1, 8632), "att-c": (12987, 11, 13014), "att-d": (12987, 3, 13006),
    "att-e": (12987, 11, 13014),
}
# the task of each model, as numpy computes it below
TASKS = {**{m: "second" for m in MODELS if m.startswith("rnn")},
         **{m: "max" for m in ("lin-a", "lin-a-1to4", "lin-b", "lin-b-4", "lin-f-4", "att-a", "att-c", "att-e")},
         **{m: "min" for m in ("lin-c", "lin-d", "lin-e", "lin-f", "att-b", "att-d")}}
QUICK = ["rnn-b", "rnn-a", "rnn-d", "rnn-e", "lin-a", "att-a", "att-b"]  # their accuracies take seconds to measure


def numpy_rnn(W, x):
    w, R, U = (W[k].astype(np.float64) for k in ("input", "recurrent", "readout"))
    h = np.maximum(x[:, :1] * w[None, :, 0], 0)
    for t in range(1, x.shape[1]):
        h = np.maximum(x[:, t : t + 1] * w[None, :, 0] + h @ R.T, 0)
    return (h @ U.T).argmax(axis=1)


def numpy_lin(W, x):
    E, H, U = (W[k].astype(np.float64) for k in ("embed", "hidden", "unembed"))
    h = E[:, x].sum(axis=2)  # (d_model, n): one embedding column per slot; the empty value's column is zeros
    return (U @ (H @ h)).T.argmax(axis=1)


def numpy_att(W, x):
    """One input at a time: attention over the slots that hold a token, read at the last of them."""
    E, P, Q, K, V, O, U = (W[k].astype(np.float64) for k in ("embed", "pos", "query", "key", "value", "out", "unembed"))
    empty, out = E.shape[0] - 1, []
    for row in x:
        k = int((row != empty).sum())
        v = (E[row] + P)[:k]
        r = v[-1]
        s = (v @ K) @ (r @ Q / math.sqrt(Q.shape[1]))
        a = np.exp(s - s.max())
        out.append(int(np.argmax((r + ((a / a.sum()) @ (v @ V)) @ O) @ U)))
    return np.array(out)


def numpy_forward(name, W, x):
    return {"rnn": numpy_rnn, "lin": numpy_lin, "att": numpy_att}[name[:3]](W, x)


def numpy_answer(m, x):
    if TASKS[m.name] == "second":
        return np.argsort(x, axis=1, kind="stable")[:, -2]
    empty = m.data_set.get("empty")
    if TASKS[m.name] == "max":
        return np.where(x == empty, -1, x).max(axis=1)
    return np.where(x == empty, 10**9, x).min(axis=1)


def draw(m, n, rng):
    """Inputs of the model's data set, drawn here without the language."""
    ds = m.data_set
    if ds["kind"] == "normal":
        return rng.standard_normal((n, m.n_ctx))
    if ds["kind"] == "all":
        return rng.integers(0, int(m.input_space[0]), size=(n, m.n_ctx))
    k = rng.choice(ds["lengths"], size=n)
    x = rng.integers(0, ds["vocab"], size=(n, m.n_ctx))
    x[np.arange(m.n_ctx)[None, :] >= k[:, None]] = ds["empty"]
    return x


@pytest.fixture(scope="module")
def rnn_g():
    return load_model("models/rnn-g")


@pytest.fixture(scope="module")
def rnn_b():
    return load_model("models/rnn-b")


def test_every_model_is_priced_and_has_its_task_here():
    assert set(MODELS) == set(PRICES) == set(TASKS) and len(MODELS) == 20


@pytest.mark.parametrize("name", MODELS)
def test_prices_of_a_forward_pass_the_answer_an_input_and_a_sample(name):
    m = load_model(f"models/{name}")
    F, A, S = PRICES[name]
    for nid in m.network_ids()[:2]:
        W = m.load_weights(nid)
        assert m.forward_ops(W) == F  # level 1's count for the same network
        assert m.answer_ops(W) == A
        assert m.sample_ops(W) == S  # one input's draws + forward pass + answer + one comparison + one more term of the sum
    ds, D = m.data_set, m.input_ops()
    varying = ds["kind"] == "lengths" and list(ds["lengths"]) != [m.n_ctx]
    # drawing an input: its numbers; with varying lengths also the length, and per slot a comparison and a where
    assert D == (3 * m.n_ctx + 2 if varying else m.n_ctx)
    assert S == D + F + A + 2


@pytest.mark.parametrize("name", MODELS)
def test_the_networks_and_the_task_agree_with_numpy(name):
    m = load_model(f"models/{name}")
    x = draw(m, 300 if name.startswith("att") else 3000, np.random.default_rng(1))
    for nid in m.network_ids():
        W = m.load_weights(nid)
        out, ans = m.outputs(W, x)
        assert np.array_equal(out, numpy_forward(name, W, x)), nid
    assert np.array_equal(ans, numpy_answer(m, x))


def test_the_accuracy_measurement_counts_what_numpy_counts(rnn_b, rnn_g):
    for m in (rnn_b, rnn_g):
        nid = m.network_ids()[0]
        W = m.load_weights(nid)
        got = m.accuracy(W, n=4096, seed=5, chunk=1024)
        # chunk k is monte_carlo(1024) run with seed [5, k]; its inputs are the first random node, stream 0
        x = np.concatenate([np.random.default_rng([5, k, 0]).standard_normal((1024, m.n_ctx)) for k in range(4)])
        assert got["method"] == "sample" and got["correct"] == int((numpy_rnn(W, x) == numpy_answer(m, x)).sum())
        assert got["accuracy"] == got["correct"] / 4096 and math.isclose(
            got["standard_error"], math.sqrt(got["accuracy"] * (1 - got["accuracy"]) / 4096))
    with pytest.raises(ValueError, match="multiple of the chunk"):
        rnn_b.accuracy(rnn_b.load_weights("rnn-b-s0"), n=1000, seed=0, chunk=300)


def test_inputs_of_varying_lengths_are_counted_as_drawn():
    m = load_model("models/lin-c")
    W = m.load_weights("lin-c-s0")
    got = m.accuracy(W, n=4096, seed=5, chunk=1024)
    # g.inputs draws the tokens (stream 0), then the number of tokens (stream 1), and empties the slots past it
    xs = []
    for k in range(4):
        tokens = np.random.default_rng([5, k, 0]).integers(0, 64, size=(1024, 4))
        length = np.random.default_rng([5, k, 1]).integers(0, 4, size=1024) + 1
        xs.append(np.where(np.arange(4)[None, :] >= length[:, None], 64, tokens))
    x = np.concatenate(xs)
    assert got["method"] == "sample" and got["correct"] == int((numpy_lin(W, x) == numpy_answer(m, x)).sum())


def test_a_data_set_that_can_be_listed_is_measured_exactly():
    m = load_model("models/lin-a")
    W = m.load_weights("lin-a-s1")
    x = np.indices([64, 64]).reshape(2, -1).T
    right = int((numpy_lin(W, x) == x.max(axis=1)).sum())
    got = m.accuracy(W, n=2**21, seed=0)
    assert got["method"] == "exact" and got["inputs"] == 4096 and got["standard_error"] == 0
    assert right == got["correct"] == 4092 and got["accuracy"] == right / 4096


def test_an_exact_measurement_weights_each_input_by_how_often_it_is_drawn(tmp_path):
    """lin-c's networks asked about one or two tokens: 64 inputs of one token and 4,096 of two, each half the data
    set, so an input of one token weighs 64 times as much as one of two."""
    shutil.copytree(REPO_ROOT / "models" / "lin-c", tmp_path / "m")
    p = tmp_path / "m" / "config.yaml"
    text = p.read_text()
    for a, b in (("n_ctx: 4", "n_ctx: 2"), ("input_space: [65, 65, 65, 65]", "input_space: [65, 65]"),
                 ("lengths: [1, 2, 3, 4]", "lengths: [1, 2]")):
        assert a in text
        text = text.replace(a, b)
    p.write_text(text)
    m = load_model(tmp_path / "m")
    W = m.load_weights("lin-c-s1")
    got = m.accuracy(W, n=8192, seed=0)
    one = np.stack([np.arange(64), np.full(64, 64)], axis=1)
    two = np.indices([64, 64]).reshape(2, -1).T
    r1, r2 = (int((numpy_lin(W, x) == numpy_answer(m, x)).sum()) for x in (one, two))
    assert got["method"] == "exact" and got["inputs"] == 4160 and got["correct"] == r1 + r2
    assert [(g["inputs"], g["correct"]) for g in got["groups"]] == [(64, r1), (4096, r2)]
    assert math.isclose(got["accuracy"], r1 / 128 + r2 / 8192) and r1 / 64 != r2 / 4096


def _recorded(name):
    return read_yaml(REPO_ROOT / "withheld" / name / "task.yaml")["accuracy"]


@pytest.mark.parametrize("name", MODELS)
def test_every_network_has_a_recorded_accuracy(name):
    assert set(_recorded(name)) == set(load_model(f"models/{name}").network_ids())


@pytest.mark.parametrize("name", QUICK)
def test_the_recorded_accuracies_are_the_verifiers_own(name):
    """task.yaml records the verifier's measurement: exact where the data set can be listed in 2^21 inputs, else on
    2^21 inputs drawn with seed 20261124. Checked here on the quick models, the others by the slow test below."""
    m, rec = load_model(f"models/{name}"), _recorded(name)
    for nid in m.network_ids():
        assert round(m.accuracy(m.load_weights(nid), n=2**21, seed=20261124)["accuracy"], 6) == rec[nid], nid


@pytest.mark.slow
@pytest.mark.parametrize("name", [m for m in MODELS if m not in QUICK])
def test_the_recorded_accuracies_of_the_other_models(name):
    m, rec = load_model(f"models/{name}"), _recorded(name)
    for nid in m.network_ids():
        assert round(m.accuracy(m.load_weights(nid), n=2**21, seed=20261124)["accuracy"], 6) == rec[nid], nid


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_rnn_weights_are_algzoos_files_converted():
    conv = _load(REPO_ROOT / "withheld" / "rnn" / "convert.py", "convert")
    assert set(conv.NETWORKS) == {m for m in MODELS if m.startswith("rnn")}
    for name, nets in conv.NETWORKS.items():
        m = load_model(f"models/{name}")
        for nid, _file in nets:
            W = m.load_weights(nid)
            w = conv.convert(name, nid)
            assert set(w) == set(W) and all(np.array_equal(w[k], W[k]) and W[k].dtype == np.float32 for k in w)


def test_the_noised_networks_are_what_the_script_makes(rnn_g):
    noise = _load(REPO_ROOT / "withheld" / "rnn-g" / "noise.py", "noise")
    first = rnn_g.load_weights("rnn-g-s0")
    for k in noise.SIGMAS:
        W = rnn_g.load_weights(f"rnn-g-n{k}")
        w = noise.noised(first, k)
        assert all(np.array_equal(w[n], W[n]) for n in W)


def test_an_attention_batch_stays_within_the_limit_per_value():
    """A batch's slot vectors are n x n_ctx x d_model numbers: max_batch keeps them under the language's limit."""
    for name in (m for m in MODELS if m.startswith("att")):
        m = load_model(f"models/{name}")
        assert m.max_batch * m.n_ctx * m.sizes["d_model"] <= lang.MAX_ELEMENTS
        assert m.effective_chunk(32768) == m.max_batch


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
