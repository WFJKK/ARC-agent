"""One session: the two tools, their budgets, what they keep, and what the agent's code is handed."""

import json

import numpy as np
import pytest

from arcagent import run_config
from arcagent.session import REFERENCES, Session, ToolError
from arcagent.util import append_jsonl, read_json, read_jsonl


BEST = {"format": "arc-estimator-v1", "nodes": [{"op": "const", "value": 0.5, "dtype": "float"}], "estimate": 0}
COPIES = {"count": 4, "runs": 8, "reference_inputs": 4096}  # the share and the size are the pilot's own


def make_session(tmp_path, copies):
    cfg = run_config.load("config/pilot/rnn-b-smoke.yaml", {"runs_dir": str(tmp_path), "sandbox": "process",
                                                           "session": {"python_calls": 3, "submissions": 2},
                                                           "verifier": {"runs": 128, "reference_inputs": 2**16, "copies": copies}})
    return Session.create(tmp_path / "run" / "round_001" / "session", cfg, "rnn-b-s0", tmp_path / "run", carried={"best": BEST})


@pytest.fixture()
def session(tmp_path):
    """A session of a run that scores on the network itself."""
    return make_session(tmp_path, {"count": 0})


@pytest.fixture()
def on_copies(tmp_path):
    """A session of a run that scores on copies of the network."""
    return make_session(tmp_path, COPIES)


CODE = """
print(sorted(info), info["forward_ops"], info["runs"], info["limits"])
print(sorted(estimators))
x = sample(1000, seed=3)
out, ans = outputs(x)
print(x.shape, out.shape, (out == ans).mean() > 0.9)
g = graph()
xs = g.inputs(16)
e = g.entry((model.forward(g, lang, xs) == task.answer(g, lang, xs)).mean())
r = run(e, runs=5, seed=1)
print(r["cost"].tolist(), r["draws"].tolist())
print(score_against(e, 0.98, runs=8)["reward"] < 0)
save_estimator("sixteen", e)
save_estimator("bad name!", e)
"""


def test_run_python_hands_the_code_what_it_needs_and_keeps_what_it_saves(session):
    out = session.run_python(CODE)
    lines = out["output"].splitlines()
    assert out["status"] == "ok", out["output"]
    assert lines[0] == ("['data_set', 'forward_ops', 'input_space', 'limits', 'max_batch', 'n_outputs', 'reference_inputs', "
                        "'runs', 'sizes', 'weights'] 78 128 {'max_nodes': 400, 'max_const_scalars': 256}")
    assert lines[1] == "['best']" and lines[2] == "(1000, 3) (1000,) True"
    assert lines[3] == f"{[16 * 89 / 78] * 5} {[48] * 5}"
    assert lines[4] == "True" and out["saved_estimators"] == ["sixteen"]
    assert "ValueError" not in out["output"] and out["problems"] == ["bad name!: not saved: an estimator's name is 1 to 40 letters, digits, - or _"]
    assert sorted(session.estimators()) == ["best", "sixteen"] and out["remaining"] == {"run_python": 2, "submit_estimator": 2}
    assert (session.path / "python" / "001.py").read_text() == CODE and (session.path / "python" / "001.txt").is_file()
    assert read_jsonl(session.path / "tools.jsonl")[0]["tool"] == "run_python"


def test_submit_estimator_returns_what_the_agent_may_see(session):
    r = session.submit_estimator("best")
    assert r["status"] == "scored" and r["runs"] == 128 and r["cost"] == 0 and not r["uses_random_draws"]
    assert "accuracy" not in json.dumps(r) and "bias" not in r and "benchmark" not in r
    assert r["remaining"] == {"run_python": 3, "submit_estimator": 1}
    folder = session.path / "submissions" / "001"
    receipt = read_json(folder / "receipt.json")
    assert receipt["name"] == "best" and receipt["parts"]["accuracy"] > 0.9 and r["reward"] == receipt["reward"]
    assert read_json(folder / "estimator.json") == session.estimators()["best"]
    with np.load(folder / "runs.npz") as z:
        assert z["estimates"].tolist() == [0.5] * 128
    assert session.receipts()[0]["submission"] == 1
    with pytest.raises(ToolError, match="no saved estimator named 'nothing'"):
        session.submit_estimator("nothing")
    session.submit_estimator("best")
    with pytest.raises(ToolError, match="budget of this session"):
        session.submit_estimator("best")


def test_the_session_reads_the_accuracy_its_run_measured(session):
    """The run folder keeps the verifier's measurement of each network; a session uses it rather than measuring
    again (and the agent never sees it)."""
    run_path = session.path.parent.parent
    append_jsonl(run_path / REFERENCES, {"network": "rnn-b-s0", "accuracy": 0.75, "correct": 49152, "n": 2**16,
                                         "seed": 20261124, "chunk": 32768, "standard_error": 0.0017})
    r = session.submit_estimator("best")
    assert r["reward"] == pytest.approx(-(0.25**2))


def test_calls_are_checked(session):
    with pytest.raises(ToolError, match="no tool"):
        session.call("read_file", {})
    with pytest.raises(ToolError, match="needs"):
        session.call("submit_estimator", {})
    with pytest.raises(ToolError, match="does not take"):
        session.call("submit_estimator", {"name": "best", "claimed": 1})
    with pytest.raises(ToolError, match="at most 200,000"):
        session.run_python("x" * 200_001)
    out = session.call("run_python", {"code": "1 / 0"})
    assert out["status"] == "ok" and "ZeroDivisionError" in out["output"]


CODE_COPIES = """
c = info["copies"]
print(sorted(c), c["count"], c["runs"], c["reference_inputs"])
w = copy(seed=3)
print(sorted(w) == sorted(weights), [bool((w[k] != weights[k]).any()) for k in sorted(w)], w["recurrent"].dtype)
same = copy(seed=3)
print(all((w[k] == same[k]).all() for k in w), any((copy(seed=4)[k] != w[k]).any() for k in w))
print(accuracy() > 0.97, accuracy(w, inputs=8192, seed=1), accuracy(inputs=8192, seed=1))
g = graph(weights=w)
print(bool((g.weight("recurrent").value == w["recurrent"]).all()))
xs = graph().inputs(16)
e = xs.g.entry((model.forward(xs.g, lang, xs) == task.answer(xs.g, lang, xs)).mean())
r = run(e, runs=5, seed=1, weights=w)
print(r["cost"].tolist(), r["estimates"].tolist() != run(e, runs=5, seed=1)["estimates"].tolist() or "same")
x = sample(2000, seed=3)
out, ans = outputs(x, weights=w)
print(out.shape, float((out == ans).mean()) != float((outputs(x)[0] == ans).mean()))
s = score_on_copies(e, copies=3, runs=4, seed=5)
print(s["reward"], s["mean_squared_error"], s["cost"], s["accuracy"].tolist(), s["estimate_mean"].tolist(), s["seeds"])
save_estimator("sixteen", e)
"""


def test_a_run_on_copies_hands_the_code_what_makes_and_measures_copies(on_copies):
    from arcagent import lang, measure

    out = on_copies.run_python(CODE_COPIES)
    lines = out["output"].splitlines()
    assert out["status"] == "ok" and len(lines) == 8, out["output"]
    assert lines[0] == "['count', 'reference_inputs', 'runs', 'share', 'size'] 4 8 4096"
    assert lines[1].startswith("True [") and lines[1].endswith("float32") and "True" in lines[1][5:]
    assert lines[2] == "True True" and lines[4] == "True"
    # a copy is the verifier's kind of copy, and its accuracy is measured as the verifier measures it
    m, v = on_copies.model(), on_copies.verifier()
    W = m.load_weights("rnn-b-s0")
    w = measure.copy_of(W, v.copies["share"], v.copies["size"], np.random.default_rng(3))
    a_copy, a_own = (m.accuracy(x, 8192, 1, 8192)["accuracy"] for x in (w, W))
    assert lines[3] == f"True {a_copy} {a_own}" and a_copy != a_own
    assert lines[5].startswith(f"{[16 * 89 / 78] * 5} ") and lines[5].endswith("True")
    assert lines[6] == "(2000,) True"
    # a scoring made in the sandbox is the verifier's own scoring for that seed
    e = on_copies.estimators()["sixteen"]
    r = v.score(e, seed=5, count=3, runs=4)
    assert lines[7] == (f"{r['reward']} {r['parts']['mse']} {r['parts']['cost']} {r['copies']['accuracy']} "
                        f"{r['copies']['estimate_mean']} {r['copies']['seeds']}")
    assert lang.entry_hash(e) == r["entry_sha256"]


def test_submit_estimator_on_copies_tells_nothing_of_the_copies_accuracies(on_copies):
    r = on_copies.submit_estimator("best")
    assert r["status"] == "scored" and r["runs"] == 32 and r["copies"] == 4 and r["runs_per_copy"] == 8
    assert r["estimate_mean"] == 0.5 and r["estimate_std"] == 0 and r["estimate_std_between_copies"] == 0
    assert "accuracy" not in json.dumps(r) and "bias" not in r and "benchmark" not in r and "tracking" not in r
    folder = on_copies.path / "submissions" / "001"
    receipt = read_json(folder / "receipt.json")
    assert receipt["scored_on"] == "copies" and len(receipt["copies"]["accuracy"]) == 4
    assert r["reward"] == receipt["reward"] == pytest.approx(-np.mean((0.5 - np.array(receipt["copies"]["accuracy"])) ** 2))
    with np.load(folder / "runs.npz") as z:
        assert z["estimates"].shape == (4, 8) and z["accuracy"].tolist() == receipt["copies"]["accuracy"]


def test_without_copies_the_helpers_for_copies_say_so(session):
    out = session.run_python("print('copies' in info)\ntry:\n    copy()\nexcept ValueError as exc:\n    print(exc)\n"
                             "try:\n    score_on_copies(estimators['best'])\nexcept ValueError as exc:\n    print(exc)")
    assert out["output"].splitlines() == ["False"] + ["this run scores an estimator on the network itself, not on copies of it"] * 2

