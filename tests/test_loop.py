"""The whole run with scripted rounds: prompts, sessions, records, stopping, resume, lock, held-out check, report."""

import math
from pathlib import Path

import numpy as np
import pytest

from arcagent import hygiene, loop, report, run_config
from arcagent.util import read_json, read_jsonl

HERE = Path(__file__).parent
FAKE = HERE / "fake_run"


def config(tmp_path, **over):
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 6,
            "session": {"python_calls": 3, "submissions": 2},
            "verifier": {"runs": 256, "reference_inputs": 2**16},
            "heldout": {"runs": 64, "check_runs": 256}}
    return run_config.load("config/pilot/rnn-b-smoke.yaml", run_config.merge(base, over))


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    lines = []
    path = loop.start(config(tmp_path_factory.mktemp("runs")), fake_scripts=FAKE, out=lines.append)
    return path, lines


def test_the_rounds_and_why_they_end(finished):
    path, lines = finished
    rounds = loop.completed(path)
    assert [rd["improved"] for rd in rounds] == [True, True, True, False, False]  # patience 2 ends it; script 6 is never asked
    assert "no higher reward in 2 rounds in a row" in next(e["ended"] for e in read_jsonl(path / "log.jsonl") if e.get("ended"))
    r1, r2, r3, r4, r5 = rounds
    A = loop.cached_references(path)["rnn-b-s0"]["accuracy"]
    assert [s["carried_as"] for s in r1["submissions"]] == ["r1s1", "r1s2"]
    s32, c09 = r1["submissions"]
    assert s32["parts"]["cost"] == pytest.approx(32 * 89 / 78) and s32["parts"]["random"]
    assert c09["reward"] == pytest.approx(-((0.9 - A) ** 2)) and not c09["parts"]["random"]
    assert r2["submissions"][0]["name"] == "sampling_128" and r2["submissions"][0]["reward"] > s32["reward"]
    # the constant 0.98, read off nothing but the agent's own sample, beats every sampler on this network
    assert r3["submissions"][0]["reward"] == pytest.approx(-((0.98 - A) ** 2)) and r3["improved"]
    assert [s["status"] for s in r4["submissions"]] == ["refused", "scored"]   # the third was over the budget
    assert "the estimate is nan" in r4["submissions"][0]["reason"] and r4["tools_used"] == {"run_python": 1, "submit_estimator": 2}
    assert r1["notes"].startswith("## What the network does") and "Some working text" not in r1["notes"]
    assert r5["notes"] == "No notes in the asked-for form this time." and r5["submissions"] == []
    assert any("WARNING: the sandbox is in process mode" in line for line in lines)
    assert any("WARNING: the reward: the reward scores only the estimate" in line for line in lines)


def test_prompts_carry_the_history_and_nothing_withheld(finished):
    path, _ = finished
    first = (path / "round_001" / "prompt.md").read_text()
    third = (path / "round_003" / "prompt.md").read_text()
    assert "This is round 1 of at most 6" in first and "None: this is the first round." in first
    assert "def forward(g, lang, x):" in first and "def answer(g, lang, x):" in first and "78 operations" in first
    assert "lambda = 1e-06" in first and "65,536 inputs" in first and "256 times" in first and "{{" not in first
    assert "second-largest" in first  # the task is told
    assert "| r2s1 | scored |" in third and "is also there as `best`" in third
    assert "    More samples, smaller error." in third  # the agent's own words come back to it
    run = read_json(path / "run.json")
    forbidden = loop.forbidden_strings(run["config"], loop.load_model(run["config"]["model"]))
    assert not hygiene.hits({"prompt": first}, forbidden)  # later prompts hold the agent's own numbers, which it knows
    for k in range(1, 6):
        text = (path / f"round_{k:03d}" / "prompt.md").read_text()
        assert "rnn-b" not in text and "withheld" not in text and "benchmark" not in text and "sampling at that cost" not in text
    assert ("| saved as | status | reward | mean squared error | cost (forward passes) | estimate: mean | estimate: spread "
            "| draws |") in third  # what the agent sees of a scoring: no accuracy, no bias
    A = loop.cached_references(path)["rnn-b-s0"]["accuracy"]
    assert f"{A:.4f}" not in first


def test_estimators_are_carried_between_rounds(finished):
    path, _ = finished
    carried = sorted(p.stem for p in (path / "round_003" / "session" / "workspace" / "estimators").glob("*.json"))
    assert carried == ["best", "constant_098", "r1s1", "r1s2", "r2s1"]  # carried from rounds 1 and 2, and saved in round 3
    assert read_json(path / "round_003" / "session" / "workspace" / "estimators" / "best.json") == \
        read_json(path / "round_002" / "session" / "submissions" / "001" / "estimator.json")


def test_every_run_of_a_scoring_is_kept(finished):
    path, _ = finished
    sub = path / "round_001" / "session" / "submissions" / "001"
    receipt = read_json(sub / "receipt.json")
    with np.load(sub / "runs.npz") as z:
        assert len(z["estimates"]) == receipt["runs"] == 256 and set(z["ops"]) == {32 * 89}
        assert z["estimates"].mean() == pytest.approx(receipt["parts"]["estimate_mean"])
    assert "runs_data" not in receipt and receipt["benchmark"]["mse_reduction"] is not None


def test_the_lock_and_the_heldout_check(finished):
    path, _ = finished
    lock, check = read_json(path / "lock.json"), read_json(path / "heldout_check.json")
    assert lock["locked"] and lock["carried_as"] == "r3s1" and lock["name"] == "constant_098"
    assert read_json(path / "locked_estimator.json") == read_json(path / "round_003" / "session" / "submissions" / "001" / "estimator.json")
    assert lock["measured_again"]["reward"] == pytest.approx(lock["reward"])  # a constant scores the same with any seed
    refs = loop.cached_references(path)
    assert set(check["networks"]) == {"rnn-b-s1", "rnn-b-s4"}
    for net, r in check["networks"].items():
        assert r["runs"] == 256 and r["entry_sha256"] == lock["entry_sha256"]
        assert r["reward"] == pytest.approx(-((0.98 - refs[net]["accuracy"]) ** 2))
    # on rnn-b-s1 the constant misses by about 9 points, far worse than the sampler it beat on the run's network
    held = {(e["carried_as"], e["network"]): e for e in read_jsonl(path / loop.HELDOUT_SCORES)}
    assert held[("r2s1", "rnn-b-s1")]["reward"] > check["networks"]["rnn-b-s1"]["reward"]
    again = loop.finish(path, out=lambda *_: None)  # from the records alone
    assert again == lock


def test_every_submission_is_scored_on_the_heldout_networks_and_the_agent_never_sees_it(finished):
    path, lines = finished
    subs = loop.scored(loop.completed(path))
    held = read_jsonl(path / loop.HELDOUT_SCORES)
    assert sorted((e["carried_as"], e["network"]) for e in held) == sorted(
        (s["carried_as"], n) for s in subs for n in ("rnn-b-s1", "rnn-b-s4"))
    assert all(e["status"] == "scored" and e["runs"] == 64 for e in held)
    assert len({e["seed"] for e in held if e["network"] == "rnn-b-s1"}) == 1  # one fixed seed per network
    assert any("on 2 held-out networks: mean reward" in line for line in lines)
    for k in range(1, 6):
        rdir = path / f"round_{k:03d}"
        assert "rnn-b-s1" not in (rdir / "round.json").read_text() and "rnn-b-s1" not in (rdir / "prompt.md").read_text()
    n = len(held)
    loop.finish(path, out=lambda *_: None)  # finishing again adds nothing
    assert len(read_jsonl(path / loop.HELDOUT_SCORES)) == n
    refs = read_jsonl(path / "references.jsonl")
    assert sorted(r["network"] for r in refs) == ["rnn-b-s0", "rnn-b-s1", "rnn-b-s4"]  # each measured once


def test_the_report(finished):
    path, _ = finished
    text = (path / "report.md").read_text()
    rep = read_json(path / "report.json")
    assert "## Submissions" in text and "## The locked estimator" in text and "### On the held-out networks" in text
    assert "## Reference estimators" in text and "plain sampling, 64 inputs on rnn-b-s0" in text
    assert "| r3s1 `constant_098` |" in text and "Mean reward on the held-out networks" in text
    assert "### Its closing notes, last round" in text and "\u2014" not in text  # no em dash
    assert "## What the agent ran" in text and "5 programs in all" in text and "The guard inside Python refused nothing." in text
    assert rep["exploration"]["reaching"] == []
    assert rep["lock"]["carried_as"] == "r3s1" and len(rep["submissions"]) == 5
    refs = read_json(path / "reference_scores.json")
    best = [r["name"] for r in refs["networks"]["rnn-b-s0"] if r["name"].startswith("plain sampling")]
    A = loop.cached_references(path)["rnn-b-s0"]["accuracy"]
    n_star = round(math.sqrt(A * (1 - A) / (1e-6 * 89 / 78)))
    assert f"plain sampling, {n_star} inputs" in best and best == sorted(best, key=lambda x: int(x.split()[2]))
    assert "| as submitted | -2." in text and "Reward `r_lambda`, lambda = 1e-06;" in text
    report.write(path)  # again, from the records
    assert (path / "report.md").read_text() == text


def test_rescore_under_another_lambda(finished):
    path, _ = finished
    rows = loop.rescore(path, {"name": "r_lambda", "lam": 0.0}, out=lambda *_: None)
    by = {r["what"]: r for r in rows}
    assert by["r1s1"]["rescored"] > by["r1s1"]["recorded"]  # compute is free now
    assert by["r3s1"]["rescored"] == pytest.approx(by["r3s1"]["recorded"])  # a constant costs nothing either way
    assert any(r["what"] == "locked on rnn-b-s1" for r in rows)


def test_a_run_stopped_by_a_usage_limit_resumes_where_it_was(tmp_path):
    cfg = config(tmp_path, run_id="resume", rounds_max=1)
    path = run_config.run_path(cfg)
    lines = []
    loop.start(cfg, fake_scripts=HERE / "fake_resume", out=lines.append)
    assert not (path / "round_001" / "round.json").exists() and not (path / "lock.json").exists()
    assert any("stopped: usage limit reached" in line for line in lines)
    assert loop.drive(path, fake_scripts=HERE / "fake_resume_after", out=lambda *_: None) == "finished"
    (rd,) = loop.completed(path)
    assert [s["name"] for s in rd["submissions"]] == ["sampling_32", "constant_09"]  # the budgets and the saved carried on
    assert rd["tools_used"] == {"run_python": 1, "submit_estimator": 2}
    assert read_json(path / "lock.json")["name"] == "sampling_32"
    with pytest.raises(RuntimeError, match="is a run already"):
        loop.start(cfg, fake_scripts=HERE / "fake_resume")


def test_a_changed_tree_is_refused_on_resume(tmp_path, monkeypatch):
    cfg = config(tmp_path, run_id="changed", rounds_max=1)
    loop.start(cfg, fake_scripts=HERE / "fake_resume", out=lambda *_: None)
    path = run_config.run_path(cfg)
    monkeypatch.setattr(loop, "trusted_hash", lambda model: "something else")
    with pytest.raises(RuntimeError, match="changed since this run began"):
        loop.drive(path, fake_scripts=HERE / "fake_resume_after", out=lambda *_: None)
    assert loop.drive(path, fake_scripts=HERE / "fake_resume_after", out=lambda *_: None, accept_changes=True) == "finished"


@pytest.mark.parametrize("pilot,exact", [("config/pilot/lin-a-smoke.yaml", 4096), ("config/pilot/att-a-smoke.yaml", 4160),
                                         ("config/pilot/att-c-smoke.yaml", None)])
def test_a_whole_run_on_networks_of_tokens(pilot, exact, tmp_path):
    """One scripted round on a linear network whose data set is listed in full, on an attention network whose data
    set is listed too though its inputs vary in length (one slot may be empty), and on one whose data set is too
    large to list: the reference, the scoring, the held-out check and the report."""
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 1,
            "session": {"python_calls": 3, "submissions": 2},
            "verifier": {"runs": 64, "reference_inputs": 2**16}, "heldout": {"runs": 16, "check_runs": 64}}
    cfg = run_config.load(pilot, base)
    lines = []
    path = loop.start(cfg, fake_scripts=FAKE, out=lines.append)
    refs = loop.cached_references(path)
    net, held = cfg["network"], cfg["heldout"]["networks"]
    assert set(refs) == {net, *held}
    A = refs[net]["accuracy"]
    if exact:
        assert all(r["method"] == "exact" and r["inputs"] == exact for r in refs.values())
        assert any(f"measured exactly, on all {exact:,} of its inputs" in line for line in lines)
        assert f"exactly, on all {exact:,} inputs of the data set" in (path / "report.md").read_text()
        told = f"runs the network on\nevery one of the data set's {exact:,} inputs, each weighted by how often it is drawn,"
    else:
        assert all(r["method"] == "sample" and r["n"] == 2**16 for r in refs.values())
        told = "runs the network on\n65,536 inputs it draws from the data set and counts"
    assert told in (path / "round_001" / "prompt.md").read_text()  # the agent is told how A is measured
    (rd,) = loop.completed(path)
    s32, c09 = rd["submissions"]
    from arcagent.networks import load_model
    m = load_model(cfg["model"])
    W = m.load_weights(net)
    assert s32["parts"]["cost"] == pytest.approx(32 * m.sample_ops(W) / m.forward_ops(W))
    assert c09["reward"] == pytest.approx(-((0.9 - A) ** 2))
    check = read_json(path / "heldout_check.json")
    assert set(check["networks"]) == set(held) and all(r["runs"] == 64 for r in check["networks"].values())
