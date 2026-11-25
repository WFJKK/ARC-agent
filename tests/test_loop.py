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
    """A run scored on the network itself (copies: count 0), as these scripted rounds were written for: there the
    number read off the network wins, and the held-out networks show it. The runs on copies are further down."""
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 6,
            "session": {"python_calls": 3, "submissions": 2},
            "verifier": {"runs": 256, "reference_inputs": 2**16, "copies": {"count": 0}},
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
    # round 4 hands in a faulty estimator and then two the run has scored before (r1s2, and the best under its
    # carried name): those two are not scored again and spend nothing
    assert [s["status"] for s in r4["submissions"]] == ["refused"]
    assert "the estimate is nan" in r4["submissions"][0]["reason"] and r4["tools_used"] == {"run_python": 1, "submit_estimator": 1}
    again = [e for e in read_jsonl(path / "round_004" / "session" / "tools.jsonl") if "already scored in this run" in str(e["outcome"].get("error"))]
    assert [e["args"]["name"] for e in again] == ["r1s2", "best"] and "round 1, submission 2" in again[0]["outcome"]["error"]
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
    assert rep["lock"]["carried_as"] == "r3s1" and len(rep["submissions"]) == 4  # round 4's two repeats were not scored
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
    large to list: the reference, the scoring on copies, the held-out check and the report."""
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 1,
            "session": {"python_calls": 3, "submissions": 2},
            "verifier": {"runs": 64, "reference_inputs": 2**16, "copies": {"count": 6, "runs": 8, "reference_inputs": 8192}},
            "heldout": {"runs": 16, "check_runs": 64}}
    cfg = run_config.load(pilot, base)
    lines = []
    path = loop.start(cfg, fake_scripts=FAKE, out=lines.append)
    refs = loop.cached_references(path)
    net, held = cfg["network"], cfg["heldout"]["networks"]
    assert set(refs) == {net, *held}
    if exact:
        assert all(r["method"] == "exact" and r["inputs"] == exact for r in refs.values())
        assert any(f"measured exactly, on all {exact:,} of its inputs" in line for line in lines)
        assert f"exactly, on all {exact:,} inputs of the data set" in (path / "report.md").read_text()
        told = f"runs the copy on\nevery one of the data set's {exact:,} inputs, each weighted by how often it is drawn,"
    else:
        assert all(r["method"] == "sample" and r["n"] == 2**16 for r in refs.values())
        told = "runs the copy on\n8,192 inputs it draws from the data set and counts"
    assert told in (path / "round_001" / "prompt.md").read_text()  # the agent is told how a copy's accuracy is measured
    (rd,) = loop.completed(path)
    s32, c09 = rd["submissions"]
    from arcagent.networks import load_model
    m = load_model(cfg["model"])
    W = m.load_weights(net)
    assert s32["parts"]["cost"] == pytest.approx(32 * m.sample_ops(W) / m.forward_ops(W)) and s32["parts"]["copies"] == 6
    receipt = read_json(path / "round_001" / "session" / "submissions" / "002" / "receipt.json")
    A = np.array(receipt["copies"]["accuracy"])
    assert receipt["reference"]["method"] == ("exact" if exact else "sample") and len(set(A.tolist())) > 1
    assert c09["reward"] == pytest.approx(-float(np.mean((0.9 - A) ** 2)))  # a number, against every copy's accuracy
    check = read_json(path / "heldout_check.json")
    assert set(check["networks"]) == set(held) and all(r["runs"] == 64 for r in check["networks"].values())
    assert all(r["scored_on"] == "network" for r in check["networks"].values())  # a held-out network, as it is


# a run scored on copies of the network ---------------------------------------------------------------
COPIES = HERE / "fake_copies"


def copies_config(tmp_path, **over):
    base = {"runs_dir": str(tmp_path), "backend": "fake", "sandbox": "process", "rounds_max": 6,
            "session": {"python_calls": 3, "submissions": 2},
            "verifier": {"runs": 256, "reference_inputs": 2**16, "copies": {"count": 32, "runs": 8, "reference_inputs": 8192}},
            "heldout": {"runs": 64, "check_runs": 256}}
    return run_config.load("config/pilot/rnn-b-smoke.yaml", run_config.merge(base, over))


@pytest.fixture(scope="module")
def on_copies(tmp_path_factory):
    lines = []
    path = loop.start(copies_config(tmp_path_factory.mktemp("copies")), fake_scripts=COPIES, out=lines.append)
    return path, lines


def test_on_copies_the_number_read_off_the_network_loses_and_a_sampler_is_locked(on_copies):
    path, lines = on_copies
    r1, r2 = loop.completed(path)
    pasted, s64 = r1["submissions"]
    (s256,) = r2["submissions"]
    A = loop.cached_references(path)["rnn-b-s0"]["accuracy"]
    assert [s["name"] for s in (pasted, s64, s256)] == ["pasted", "sampling_64", "sampling_256"]
    assert pasted["parts"]["estimate_mean"] == pytest.approx(A, abs=0.01) and pasted["parts"]["cost"] == 0  # its own 8,192 inputs
    # the number is far behind: behind both samplers, and at more than three times the loss of the best plain
    # sampling on the very copies it was scored on
    assert pasted["reward"] < s64["reward"] < s256["reward"] < 0 and pasted["reward"] < 3 * pasted["benchmark"]["best_reward"]
    assert pasted["parts"]["tracking"] is None and s64["parts"]["tracking"] > 0.8
    assert pasted["benchmark"]["constant_unchanged"] == pytest.approx(pasted["reward"], rel=0.25)
    assert [rd["improved"] for rd in (r1, r2)] == [True, True]
    lock = read_json(path / "lock.json")
    assert lock["name"] == "sampling_256" and lock["parts"]["copies"] == 32 and lock["runs"] == 256
    assert lock["measured_again"]["parts"]["copies"] == 32 and lock["measured_again"]["seed"] != lock["seed"]
    alone = lock["on_the_network"]
    assert alone["status"] == "scored" and "copies" not in alone["parts"] and alone["runs"] == 256
    assert any("every submission scored on the run's own 32 copies of the network" in line for line in lines)
    seeds = {s["seed"] for rd in (r1, r2) for s in rd["submissions"]}
    assert seeds == {read_json(path / "scoring_seed.json")["seed"]} and lock["measured_again"]["seed"] not in seeds
    assert not any("WARNING: the reward" in line for line in lines)  # the warning of a run on the network itself
    assert any("the locked estimator on the network itself: reward" in line for line in lines)
    assert loop.finish(path, out=lambda *_: None) == lock  # from the records alone: nothing is measured again


def test_on_copies_the_agent_is_told_how_copies_are_made_and_nothing_of_their_accuracies(on_copies):
    path, _ = on_copies
    first = (path / "round_001" / "prompt.md").read_text()
    for said in ("The estimator is not scored on the network as you hold it but on copies of it",
                 "It scores it on\n32 copies of the network, each with a few of its weights changed at random",
                 "The verifier draws its copies once for this run, from a seed it keeps to itself",
                 "an estimator that has been scored is not scored again (the call is refused and costs no\nsubmission)",
                 "the best estimator is scored once more on fresh copies",
                 "Every weight is changed with probability 0.125, each on its own",
                 "standard deviation is 0.0398\ntimes the root mean square of the entries of its weight matrix",
                 "8,192 inputs it draws from the data set", "estimator 8 times on the copy", "That is 256 runs in all.",
                 "s = score_on_copies(e, copies=16)", "`copy(seed)` in your sandbox makes a\ncopy this way"):
        assert said in first, said
    second = (path / "round_002" / "prompt.md").read_text()
    assert "| estimate: spread on a copy | estimate: spread between copies | draws |" in second
    assert "| r1s1 | scored | " in second and " | 0.00e+00 | 0.00e+00 | no |" in second  # the number: no spread at all
    A = loop.cached_references(path)["rnn-b-s0"]["accuracy"]
    forbidden = loop.forbidden_strings(read_json(path / "run.json")["config"], __import__("arcagent.networks", fromlist=["x"]).load_model("models/rnn-b"))
    for k in (1, 2):
        rdir = path / f"round_{k:03d}"
        seen = [s["seen"] for s in read_json(rdir / "round.json")["submissions"]]
        assert all("accuracy" not in s and "tracking" not in s and "estimate_std_between_copies" in s for s in seen)
        assert "benchmark" not in (rdir / "prompt.md").read_text()
    assert not hygiene.hits({"prompt": first.split("## Earlier rounds")[0]}, forbidden)
    assert f"{A:.3f}" not in first and "rnn-b" not in first


def test_on_copies_the_heldout_networks_are_scored_as_they_are(on_copies):
    path, _ = on_copies
    held = read_jsonl(path / loop.HELDOUT_SCORES)
    assert len(held) == 6 and all(e["status"] == "scored" and "copies" not in e["parts"] and e["runs"] == 64 for e in held)
    refs = loop.cached_references(path)
    by = {(e["name"], e["network"]): e for e in held}
    for net in ("rnn-b-s1", "rnn-b-s4"):
        # the number read off the run's network, on another network: its error is the distance of the two accuracies
        assert by[("pasted", net)]["parts"]["bias"] == pytest.approx(refs["rnn-b-s0"]["accuracy"] - refs[net]["accuracy"], abs=0.01)
    check = read_json(path / "heldout_check.json")
    assert all(r["scored_on"] == "network" and r["runs"] == 256 for r in check["networks"].values())
    # and on copies of each held-out network: the locked sampler does there what it does on the run's copies
    lock, text = read_json(path / "lock.json"), (path / "report.md").read_text()
    assert set(check["copies"]) == set(check["networks"]) and "### On copies of the held-out networks" in text
    for net, r in check["copies"].items():
        assert r["scored_on"] == "copies" and r["network"] == net and r["parts"]["copies"] == 32 and f"| copies of {net} |" in text
        assert r["seed"] not in (lock["seed"], lock["measured_again"]["seed"]) and r["parts"]["tracking"] > 0.8


def test_on_copies_the_report(on_copies):
    path, _ = on_copies
    text = (path / "report.md").read_text()
    for said in ("- Scored on copies: the verifier makes 32 copies of the network once for the run, each weight changed "
                 "with probability 0.125 by a normal draw of 0.0398 times the root mean square of its matrix",
                 "an estimator once scored is not scored again", "it is the figure to quote.",
                 "| follows the copies |", "| scored again, fresh seed and fresh copies |", "| on the network itself |",
                 "What a number written into an estimator would score on the copies of the second scoring:",
                 "| the network's own accuracy, written in as a constant on rnn-b-s0 (copies) |",
                 "`rnn-b-s0` is scored on copies, as in the rounds; the held-out networks are scored as they are."):
        assert said in text, said
    assert "\u2014" not in text
    refs = read_json(path / "reference_scores.json")["networks"]
    own = {r["name"]: r for r in refs["rnn-b-s0"]}
    pasted = own["the network's own accuracy, written in as a constant"]
    best = [n for n in own if n.startswith("plain sampling")]
    n_star = round(pasted["benchmark"]["best_samples"])
    assert f"plain sampling, {n_star} inputs" in best and all(own[n]["scored_on"] == "copies" for n in best)
    assert pasted["reward"] < 4 * own[f"plain sampling, {n_star} inputs"]["reward"]
    assert all(r["scored_on"] == "network" for r in refs["rnn-b-s1"])
    report.write(path)  # again, from the records
    assert (path / "report.md").read_text() == text
    rows = loop.rescore(path, {"name": "r_lambda", "lam": 0.0}, out=lambda *_: None)
    assert {"locked, fresh seed", "locked, on the network itself"} <= {r["what"] for r in rows}
