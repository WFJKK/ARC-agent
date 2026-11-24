"""One session: the two tools, their budgets, what they keep, and what the agent's code is handed."""

import json

import numpy as np
import pytest

from arcagent import run_config
from arcagent.session import REFERENCES, Session, ToolError
from arcagent.util import append_jsonl, read_json, read_jsonl


@pytest.fixture()
def session(tmp_path):
    cfg = run_config.load("config/pilot/rnn-b-smoke.yaml", {"runs_dir": str(tmp_path), "sandbox": "process",
                                                           "session": {"python_calls": 3, "submissions": 2},
                                                           "verifier": {"runs": 128, "reference_inputs": 2**16}})
    return Session.create(tmp_path / "run" / "round_001" / "session", cfg, "rnn-b-s0", tmp_path / "run",
                          carried={"best": {"format": "arc-estimator-v1", "nodes": [{"op": "const", "value": 0.5, "dtype": "float"}],
                                            "estimate": 0}})


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
    assert lines[0] == ("['data_set', 'forward_ops', 'input_space', 'limits', 'n_outputs', 'reference_inputs', 'runs', 'sizes', "
                        "'weights'] 78 128 {'max_nodes': 400, 'max_const_scalars': 256}")
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
