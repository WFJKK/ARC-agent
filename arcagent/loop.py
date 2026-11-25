"""The loop: the rounds on one network, then the lock, the held-out check and the report.

    python -m arcagent.loop run --config config/pilot/rnn-g.yaml [--run-id X] [--sandbox hardened] [--fake DIR]
                                [--auth api_key] [--budget-usd 60]
                              [--backend codex|openai_api --agent-model M --run-id ID]
    python -m arcagent.loop resume --run <run folder> [--budget-usd 80]
    python -m arcagent.loop finish --run <run folder>    the lock, the held-out check and the report again, from the records
    python -m arcagent.loop rescore --run <run folder> --lam 1e-7    the records under another reward
                                (--param KEY=VALUE for any other parameter of a reward)

A run folder (outside the repository, <runs_dir>/<run_id>):

    run.json              the config, the commit, the hash of the trusted files, the sandbox and its probe
    log.jsonl             what happened, appended
    references.jsonl      the verifier's measurement of each network's accuracy, appended once per network
    spend.jsonl           a run billed to an API key: every try's cap and cost, appended (budget.py)
    round_001/            prompt.md, session/ (session.py), response.md, meta.json, round.json
    heldout_scores.jsonl  every scored submission also scored on the held-out networks, appended after each round;
                          nothing the agent sees reads it, and nothing is chosen by it
    locked_estimator.json the best estimator of the rounds, as submitted
    lock.json             which submission it is, its hash, its receipt, a second scoring with a fresh seed (and
                          fresh copies) and, for a run scored on copies, one scoring on the network itself
    heldout_check.json    the locked estimator on every held-out network, with the full number of runs, and (a run
                          scored on copies) on copies of every held-out network
    reference_scores.json the withheld reference estimators and plain sampling, scored under the run's conditions
    report.md, report.json

A round is done when its round.json is written. `resume` starts at the first round that is not, and a session that
was cut off keeps its budgets and submissions. The rounds end at rounds_max, after `patience` rounds in a row that
did not raise the best reward, after max_consecutive_refusals refusals in a row, or, billed to an API key, when the
run's budget is spent (a round that was under way is ended and kept first).

When they end, the submission with the highest reward is locked: the estimator as it was submitted, byte for byte
(its hash is checked). It is scored once more with a fresh seed, because the best of several noisy scorings leans
high, and then on every held-out network. No agent takes part and nothing of the estimator can change, so what comes
out is how an estimator written on one network does on networks it was not written on.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

from . import budget, hygiene, lang, prompt, rewards, run_config
from .backends import StopRun, make
from .networks import load_model
from .session import REFERENCES, Session, cached_references, surface
from .util import REPO_ROOT, append_jsonl, atomic_write, git_state, now, read_json, read_jsonl, sha256_file, write_json
from .verifier import Verifier, agent_view, describe

TRUSTED_GLOBS = ("arcagent/**/*.py", "prompts/*.md", "agent_docs/*.md")
NOTES_HEADING = "## What the network does"
MAX_NOTES = 6000
HELDOUT_SCORES = "heldout_scores.jsonl"


def trusted_hash(model) -> str:
    """One hash over the code, the prompt, the agent's guide, the model folder and the withheld folder."""
    files = [p for g in TRUSTED_GLOBS for p in REPO_ROOT.glob(g)]
    for root in (model.path, model.withheld()):
        if root.is_dir():
            files += list(root.rglob("*"))
    h = hashlib.sha256()
    for p in sorted({q.resolve() for q in files if q.is_file() and "__pycache__" not in q.parts}):
        h.update(os.path.relpath(p, REPO_ROOT).encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\0")
    return h.hexdigest()


class Log:
    def __init__(self, path: Path, out=print):
        self.path, self.out = Path(path), out

    def __call__(self, message: str, **fields) -> None:
        self.out(message)
        append_jsonl(self.path / "log.jsonl", {"at": now(), "message": message, **fields})


# what the records say ---------------------------------------------------------------------------
def completed(path: Path) -> list:
    """The rounds that are done, in order: round.json of round 1, 2, ... until one is missing."""
    out = []
    while True:
        f = Path(path) / f"round_{len(out) + 1:03d}" / "round.json"
        if not f.is_file():
            return out
        out.append(read_json(f))


def scored(rounds: list) -> list:
    return [s for rd in rounds for s in rd["submissions"] if s["status"] == "scored"]


def best_submission(rounds: list) -> dict | None:
    """The scored submission with the highest reward; the earliest one on a tie."""
    best = None
    for s in scored(rounds):
        if best is None or s["reward"] > best["reward"]:
            best = s
    return best


def stop_reason(rounds: list, cfg: dict) -> str | None:
    if len(rounds) >= int(cfg["rounds_max"]):
        return f"rounds_max ({cfg['rounds_max']}) reached"
    tail = 0
    for rd in reversed(rounds):
        if rd["improved"]:
            break
        tail += 1
    if tail >= int(cfg["patience"]):
        return f"no higher reward in {tail} rounds in a row (patience {cfg['patience']})"
    refusals = 0
    for rd in reversed(rounds):
        if rd.get("stop_reason") != "refusal":
            break
        refusals += 1
    if refusals >= int(cfg["max_consecutive_refusals"]):
        return f"the model declined {refusals} rounds in a row"
    return None


def estimator_of(path: Path, sub: dict) -> dict:
    return read_json(Path(path) / f"round_{sub['round']:03d}" / "session" / "submissions" / f"{sub['submission']:03d}" / "estimator.json")


def carried_estimators(path: Path, rounds: list) -> dict:
    """The scored estimators of earlier rounds, by the names the prompt's history gives them, and the best as `best`."""
    out = {s["carried_as"]: estimator_of(path, s) for s in scored(rounds)}
    best = best_submission(rounds)
    if best is not None:
        out["best"] = estimator_of(path, best)
    return out


def notes_of(text: str) -> str:
    """The agent's closing notes: from the first asked-for heading on, or the end of what it wrote."""
    i = text.rfind(NOTES_HEADING)
    return (text[i:] if i >= 0 else text[-MAX_NOTES:])[:MAX_NOTES].strip()


# the networks' accuracies, measured once per run folder -------------------------------------------
def verifier_for(path: Path, cfg: dict, model, network: str, log=None) -> Verifier:
    """A verifier on one network, with the accuracy the run folder keeps for it, or measured now and kept."""
    v = Verifier(model, network, cfg)
    ref = cached_references(path).get(network)
    if ref is not None:
        try:
            v.set_reference(ref)
            return v
        except ValueError:
            pass  # kept under other settings; the newest line counts
    r = v.reference()
    append_jsonl(Path(path) / REFERENCES, {**r, "at": now()})
    if log:
        how = f"exactly, on all {r['inputs']:,} of its inputs" if r.get("method") == "exact" else f"on {r['n']:,} inputs"
        log(f"  the accuracy of {network}: measured {how} (kept in {REFERENCES}, never shown to the agent)")
    return v


def heldout_networks(cfg: dict) -> list:
    return [str(n) for n in ((cfg.get("heldout") or {}).get("networks") or [])]


def heldout_seed(run_id: str, network: str) -> int:
    """The one seed with which every submission of a run is scored on a held-out network, so that they can be
    compared with each other without the noise of separate seeds."""
    return int.from_bytes(hashlib.sha256(f"{run_id}: every submission on {network}".encode()).digest()[:8], "big")


def heldout_scores(path: Path, cfg: dict, model, rounds: list, log=None) -> list:
    """Score every scored submission of these rounds on every held-out network, unless heldout_scores.jsonl has it
    already, and append it there. The agent never sees these scores (the prompt reads round.json), and nothing is
    chosen by them: the lock stays the best submission on the run's network."""
    path = Path(path)
    nets = heldout_networks(cfg)
    if not nets or not (cfg.get("heldout") or {}).get("every_submission", True):
        return []
    have = {(e.get("carried_as"), e.get("entry_sha256"), e.get("network")) for e in read_jsonl(path / HELDOUT_SCORES)}
    out = []
    runs = int(cfg["heldout"]["runs"])
    for net in nets:
        todo = [s for s in scored(rounds) if (s["carried_as"], s["entry_sha256"], net) not in have]
        if not todo:
            continue
        v = verifier_for(path, cfg, model, net, log)
        for s in todo:
            est = estimator_of(path, s)
            if lang.entry_hash(est) != s["entry_sha256"]:
                raise RuntimeError(f"{s['carried_as']}'s estimator on disk is not the one the verifier scored")
            # a held-out network is another network: the estimator is scored on it as it is, not on copies of it
            r = v.score(est, seed=heldout_seed(cfg["run_id"], net), runs=runs, copies=False)
            line = {"round": s["round"], "submission": s["submission"], "carried_as": s["carried_as"], "name": s["name"],
                    "entry_sha256": s["entry_sha256"], "network": net, "status": r["status"], "reason": r["reason"],
                    "reward": r["reward"], "parts": r["parts"], "seed": r["seed"], "runs": r["runs"], "at": now()}
            append_jsonl(path / HELDOUT_SCORES, line)
            out.append(line)
    if log and out:
        per = {}
        for line in out:
            per.setdefault(line["carried_as"], []).append(line)
        for name, lines in per.items():
            ok = [x for x in lines if x["status"] == "scored"]
            if ok:
                mean = sum(x["reward"] for x in ok) / len(ok)
                rms = (sum(x["parts"]["mse"] for x in ok) / len(ok)) ** 0.5
                log(f"  {name} on {len(ok)} held-out networks: mean reward {mean:.4e}, rms error {100 * rms:.3f} pp")
    return out


# a run ---------------------------------------------------------------------------------------------
def forbidden_strings(cfg: dict, model) -> list:
    return sorted({str(s) for s in list(cfg.get("hygiene") or []) + model.hygiene()})


def agent_surface(cfg: dict, model, forward_ops: int, sandbox_mode: str | None = None) -> dict:
    """Every text the pipeline itself puts within the agent's reach, besides a round's prompt: the session's surface
    (session.py), the note a restarted session gets, and the paths that can show from inside: the temporary folder
    the CLI is started in and, unless the agent's code runs in a container (which has its own paths), the sandbox
    folder's location and this Python's (a traceback prints it)."""
    out = {**surface(model, forward_ops, cfg), "restart note": prompt.RESTART_NOTE,
           "temporary folder's location": tempfile.gettempdir()}
    if sandbox_mode != "container":
        out["sandbox folder's location"] = str(cfg.get("sandbox_dir") or tempfile.gettempdir())
        out["python's location"] = " ".join(sorted({sys.prefix, sys.base_prefix, sys.executable,
                                                     str(cfg.get("python_executable") or "")}))
    return out


def start(cfg: dict, fake_scripts=None, out=print) -> Path:
    """Make the run folder, check everything that can be checked before a model is asked, then run the rounds."""
    from .sandbox import check_probe, choose, probe, protected_paths

    path = run_config.run_path(cfg)
    if (path / "run.json").exists():
        raise RuntimeError(f"{path} is a run already: resume it, or give the config another run_id")
    model = load_model(cfg["model"])
    model.network(cfg["network"])
    for net in heldout_networks(cfg):
        model.network(net)
        if net == cfg["network"]:
            raise RuntimeError("a held-out network must be another network than the rounds'")
    make(cfg, log=out, fake_scripts=fake_scripts)  # a missing CLI or login fails here, before a run folder exists
    path.mkdir(parents=True, exist_ok=True)
    log = Log(path, out)
    verifier = verifier_for(path, cfg, model, cfg["network"], log)
    # Plain sampling must come out at what it draws: n inputs at one sampled input's price each.
    probe_runs = verifier.score(model.monte_carlo(verifier.weights, 4), runs=8, copies=False)
    if probe_runs["status"] != "scored" or abs(probe_runs["parts"]["cost"] * verifier.F - 4 * verifier.sample_ops) > 1e-6:
        raise RuntimeError(f"plain sampling written as an estimator does not cost what it draws: {describe(probe_runs)}")
    sb = choose(cfg, protected_paths(cfg, model, path))
    first_prompt = prompt.build(cfg, model, 1, [], verifier.F)
    hygiene.check({"prompt": first_prompt, **agent_surface(cfg, model, verifier.F, sb.mode)}, forbidden_strings(cfg, model))
    report = probe(sb, model, cfg["network"], path / "probe")                      # the system's sandbox on its own
    guarded = probe(sb, model, cfg["network"], path / "probe", guard=True)         # with the guard inside Python
    try:
        (path / "probe").rmdir()  # the probe's target folder; it leaves nothing in it
    except OSError:
        pass
    check_probe(sb, report)
    check_probe(sb, guarded)
    for w in rewards.warnings(cfg["reward"], on_copies=verifier.copies is not None):
        log(f"WARNING: the reward: {w}")
    if not sb.hardened:
        log("WARNING: the sandbox is in process mode: only a guard inside Python keeps the agent's code from your files, "
            "the withheld records and other programs. It blocked every escape the probe tried, and records every attempt, "
            "but it shares its process with the code it watches: a fence, not a wall. A run that counts needs container, "
            "landlock or seatbelt mode.")
    ref = verifier.reference()
    write_json(path / "run.json", {
        "run_id": cfg["run_id"], "config": cfg, "created": now(), "git": git_state(), "trusted_hash": trusted_hash(model),
        "sandbox": {**sb.describe(), "probe": report, "guard_probe": guarded}, "prices": lang.PRICES, "format": lang.FORMAT,
        "forward_ops": verifier.F, "sample_ops": verifier.sample_ops,
        "reference": {k: ref[k] for k in ("n", "seed", "chunk", "standard_error")},
    })
    c = verifier.copies
    how = (f"every submission scored on the run's own {c['count']:,} copies of the network (each weight changed with "
           f"probability {c['share']:g}, by {c['size']:g} times its matrix's RMS; the lock once more on fresh ones), "
           f"{c['runs']:,} runs on each" if c
           else f"{int(cfg['verifier']['runs']):,} runs per scoring, the same seeds for every submission")
    log(f"run {cfg['run_id']} at {path}: {cfg['model']} / {cfg['network']}, sandbox {sb.mode}, backend {cfg['backend']}; "
        f"a forward pass is {verifier.F:,} operations; {how}; {len(heldout_networks(cfg))} held-out networks")
    if cfg["backend"] in ("claude_code", "openai_api") and cfg["auth"] == "api_key":
        b = cfg["budget"]
        count = ("by Claude Code's count" if cfg["backend"] == "claude_code"
                 else f"counted from the tokens OpenAI's API reports at the config's prices for {cfg['agent_model']}")
        log(f"billed to the API key in {cfg['api_key_env']}: at most ${float(b['run_usd']):.2f} for the run and "
            f"${float(b['round_usd']):.2f} a round, {count} (spend.jsonl)")
    elif cfg["backend"] == "claude_code":
        log("under your Claude Code login: no API key is used")
    elif cfg["backend"] == "codex":
        log("under your Codex (ChatGPT) login: no API key is used, and the CLI is checked to offer the model the session's "
            "tools and nothing that reads a file or the web before the first round")
    drive(path, fake_scripts=fake_scripts, out=out)
    return path


def drive(path: Path, fake_scripts=None, out=print, accept_changes: bool = False, budget_usd: float | None = None) -> str:
    """Run the rounds that are left, then finish. Returns how it ended: finished, or stopped (resume later).
    budget_usd: a new budget for a run billed to an API key, kept in its spend log."""
    path = Path(path)
    run = read_json(path / "run.json")
    cfg = run["config"]
    log = Log(path, out)
    if budget_usd is not None:
        if cfg.get("auth") != "api_key":
            raise RuntimeError("this run is not billed to an API key, so it has no budget to change")
        errors = budget.check({**cfg["budget"], "run_usd": budget_usd})
        if errors:
            raise RuntimeError("; ".join(errors))
        before = budget.state(path, cfg["budget"])
        budget.set_run_budget(path, budget_usd)
        log(f"the run's API budget is now ${budget_usd:.2f} (was ${before['run_usd']:.2f}; ${before['spent']:.2f} counted so far)",
            budget_usd=budget_usd)
    model = load_model(cfg["model"])
    if trusted_hash(model) != run["trusted_hash"]:
        if not accept_changes:
            raise RuntimeError("the code, the prompt, the guide or the model folder changed since this run began; "
                               "resume with --accept-changes to go on all the same (the log will say so)")
        log("the trusted files changed since the run began; going on as asked", trusted_hash=trusted_hash(model))
    backend = make(cfg, log=log, fake_scripts=fake_scripts)
    forbidden = forbidden_strings(cfg, model)
    while True:
        done = completed(path)
        why = stop_reason(done, cfg)
        if why is None and not backend.available(len(done)):
            why = "the backend has no answer for another round"
        # A spent budget ends the rounds; a round that was under way runs first, and ends at once with what it
        # submitted, so that its submissions are kept.
        if why is None and hasattr(backend, "exhausted") and not (path / f"round_{len(done) + 1:03d}").exists():
            why = backend.exhausted(path)
        if why is not None:
            log(f"the rounds end: {why}", ended=why)
            break
        try:
            run_round(path, cfg, model, len(done) + 1, done, backend, forbidden, run, log)
        except StopRun as exc:
            log(f"stopped: {exc}. Continue with: python -m arcagent.loop resume --run {path}", stopped=str(exc))
            return "stopped"
    finish(path, out=out)
    return "finished"


def run_round(path: Path, cfg: dict, model, k: int, done: list, backend, forbidden: list, run: dict, log) -> dict:
    rdir = path / f"round_{k:03d}"
    session = Session.create(rdir / "session", cfg, cfg["network"], path, carried_estimators(path, done))
    text = prompt.build(cfg, model, k, done, run["forward_ops"])
    # What the pipeline itself says is checked; the history is the agent's own words about what it found.
    ours = prompt.build(cfg, model, k, [], run["forward_ops"])
    hygiene.check({"prompt": ours, **agent_surface(cfg, model, run["forward_ops"], run["sandbox"]["mode"])}, forbidden)
    atomic_write(rdir / "prompt.md", text)
    log(f"round {k}: asking {backend.name}")
    response = backend.respond(text, k - 1, rdir, session)
    atomic_write(rdir / "response.md", response.text)
    write_json(rdir / "meta.json", response.meta)
    session.reload()
    subs = []
    for r in session.receipts():
        subs.append({"round": k, "submission": r["submission"], "name": r.get("name"), "carried_as": f"r{k}s{r['submission']}",
                     "status": r["status"], "reason": r["reason"], "reward": r["reward"], "parts": r["parts"], "size": r["size"],
                     "entry_sha256": r["entry_sha256"], "seed": r["seed"], "runs": r["runs"], "benchmark": r.get("benchmark"),
                     "seen": agent_view(r)})
    before = best_submission(done)
    got = [s["reward"] for s in subs if s["status"] == "scored"]
    improved = bool(got) and (before is None or max(got) > before["reward"] + 1e-15)
    record = {"round": k, "submissions": subs, "improved": improved, "best_reward_before": None if before is None else before["reward"],
              "best_reward_after": max(got + ([before["reward"]] if before else [])) if (got or before) else None,
              "stop_reason": response.meta.get("stop_reason"), "notes": notes_of(response.text), "tools_used": dict(session.state["used"]),
              "at": now()}
    write_json(rdir / "round.json", record)
    for s in subs:
        log(f"  {s['carried_as']} ({s['name']}): " + (describe(s) if s["status"] == "scored" else f"refused: {s['reason']}"))
    best_text = "none" if record["best_reward_after"] is None else f"{record['best_reward_after']:.6e}"
    log(f"round {k} done: {len(subs)} submissions, best reward so far {best_text}", round=k, improved=improved)
    try:  # for the record only; a failure here must not cost the run, and finish tries again
        heldout_scores(path, cfg, model, [record], log)
    except Exception as exc:
        log(f"  could not score round {k}'s submissions on the held-out networks ({type(exc).__name__}: {exc}); finish tries again")
    return record


def reference_scores(path: Path, cfg: dict, model, log=None) -> dict:
    """The withheld reference estimators on every network of the run, and plain sampling at a few sizes on the run's
    network, under the run's conditions. Measured once and kept (reference_scores.json); finishing again reuses them."""
    path = Path(path)
    f = path / "reference_scores.json"
    if f.is_file():
        return read_json(f)
    seed = int.from_bytes(hashlib.sha256(f"{cfg['run_id']}: the references".encode()).digest()[:8], "big")
    out = {"seed": seed, "networks": {}}
    for net in [cfg["network"]] + heldout_networks(cfg):
        v = verifier_for(path, cfg, model, net, log)
        own = net == cfg["network"]
        on_copies = None if own else False  # the run's network as the rounds scored it; a held-out network as it is
        rows = []
        for name, est in model.reference_entries(v.weights).items():
            rows.append({"name": name, **v.score(est, seed=seed, copies=on_copies)})
        if own:
            lam = rewards.lam_of(cfg["reward"])
            sizes = [int(n) for n in (cfg.get("report") or {}).get("sampling_sizes", [16, 64, 256])]
            A = v.reference()["accuracy"]
            if v.copies is not None:
                # what the rounds are for: the network's own accuracy, written in as a constant, on the copies
                g = model.graph(v.weights)
                pasted = v.score(g.entry(g.const(float(A))), seed=seed)
                rows.append({"name": "the network's own accuracy, written in as a constant", **pasted})
                var = (pasted.get("benchmark") or {}).get("variance") or A * (1 - A)
            else:
                var = A * (1 - A)
            if lam:
                best = max(1, round((var / (lam * v.sample_ops / v.F)) ** 0.5))
                sizes = sorted(set(sizes + [best]))
            for n in sizes:
                rows.append({"name": f"plain sampling, {n} inputs", **v.score(model.monte_carlo(v.weights, n), seed=seed)})
        out["networks"][net] = rows
    write_json(f, out)
    return out


def finish(path: Path, out=print) -> dict:
    """Lock the best estimator, score it again and on the held-out networks, write the report. Works from the records
    alone, so it can be run again."""
    from . import report

    path = Path(path)
    cfg = read_json(path / "run.json")["config"]
    log = Log(path, out)
    model = load_model(cfg["model"])
    done = completed(path)
    best = best_submission(done)
    # What an earlier finish measured is a result and is kept: finishing again never measures it anew.
    old_lock = read_json(path / "lock.json") if (path / "lock.json").is_file() else {}
    old_check = read_json(path / "heldout_check.json") if (path / "heldout_check.json").is_file() else {}
    heldout_scores(path, cfg, model, done, log)  # any a round could not do
    if best is None:
        lock = {"locked": False, "reason": "no submission was scored"}
        write_json(path / "lock.json", lock)
        log("nothing to lock: no submission was scored")
    else:
        est = estimator_of(path, best)
        if lang.entry_hash(est) != best["entry_sha256"]:
            raise RuntimeError("the best submission's estimator on disk is not the one the verifier scored")
        write_json(path / "locked_estimator.json", est)
        lock = {"locked": True, "entry_sha256": best["entry_sha256"], "network": cfg["network"],
                **{k: best[k] for k in ("round", "submission", "name", "carried_as", "status", "reward", "parts", "size", "seed",
                                        "runs", "benchmark")}}
        if old_lock.get("entry_sha256") == lock["entry_sha256"] and old_lock.get("measured_again"):
            lock["measured_again"] = old_lock["measured_again"]
        else:
            # The lock is the best of the run's submissions on the run's own seed (its copies, its draws): chosen
            # there, its recorded reward leans high. One more scoring with a fresh seed is a fair measurement of it.
            again = verifier_for(path, cfg, model, cfg["network"], log).score(est)
            if again["entry_sha256"] != lock["entry_sha256"]:
                raise RuntimeError("the second scoring did not score the locked estimator")
            lock["measured_again"] = {k: again.get(k) for k in ("status", "reason", "reward", "parts", "seed", "runs", "benchmark")}
        v = verifier_for(path, cfg, model, cfg["network"], log)
        if v.copies is not None:
            # The rounds scored it on copies. Once, for the report, on the network itself: where a number written
            # into an estimator would have won.
            if old_lock.get("entry_sha256") == lock["entry_sha256"] and old_lock.get("on_the_network"):
                lock["on_the_network"] = old_lock["on_the_network"]
            else:
                alone = v.score(est, copies=False)
                lock["on_the_network"] = {k: alone.get(k) for k in ("status", "reason", "reward", "parts", "seed", "runs", "benchmark")}
        write_json(path / "lock.json", lock)
        log(f"locked {best['carried_as']} ({best['entry_sha256'][:12]}): " + describe(best))
        log(f"the locked estimator with a fresh seed: {describe(lock['measured_again'])}")
        if lock.get("on_the_network"):
            log(f"the locked estimator on the network itself: {describe(lock['on_the_network'])}")
        nets = heldout_networks(cfg)
        if nets:
            on_copies = verifier_for(path, cfg, model, cfg["network"], log).copies is not None
            if (old_check.get("entry_sha256") == lock["entry_sha256"] and set(old_check.get("networks") or {}) == set(nets)
                    and (not on_copies or set(old_check.get("copies") or {}) == set(nets))):
                check = old_check
            else:
                check = {"entry_sha256": lock["entry_sha256"], "at": now(), "networks": {}}
                for net in nets:
                    r = verifier_for(path, cfg, model, net, log).score(est, runs=int(cfg["heldout"]["check_runs"]), copies=False)
                    if r["entry_sha256"] != lock["entry_sha256"]:
                        raise RuntimeError("the held-out check did not score the locked estimator")
                    check["networks"][net] = r
                if on_copies:
                    # And on copies of each held-out network, made as the run's own are: a held-out network as it is
                    # may be right on nearly every input and then says little, and an estimator fitted to the copies
                    # of the run's network (a layout, a threshold, a factor) shows here, on copies of another.
                    check["copies"] = {}
                    for net in nets:
                        r = verifier_for(path, cfg, model, net, log).score(est)
                        if r["entry_sha256"] != lock["entry_sha256"]:
                            raise RuntimeError("the held-out check did not score the locked estimator")
                        check["copies"][net] = r
                write_json(path / "heldout_check.json", check)
            for net, r in check["networks"].items():
                log(f"  held out, {net}: {describe(r)}")
            for net, r in (check.get("copies") or {}).items():
                log(f"  held out, copies of {net}: {describe(r)}")
    try:
        reference_scores(path, cfg, model, log)
    except Exception as exc:  # for the report only
        log(f"the reference estimators could not be scored ({type(exc).__name__}: {exc})")
    report.write(path)
    log(f"report: {path / 'report.md'}")
    return lock


def rescore(path: Path, reward_cfg: dict, out=print) -> list:
    """Every scored submission, the second scoring of the lock and the held-out check under another reward, from the
    parts on record."""
    path = Path(path)
    rows = []
    for s in scored(completed(path)):
        r = rewards.compute(s["parts"], reward_cfg)
        rows.append({"what": s["carried_as"], "name": s["name"], "recorded": s["reward"], "rescored": r["reward"]})
    lock = read_json(path / "lock.json") if (path / "lock.json").is_file() else {}
    for key, what in (("measured_again", "locked, fresh seed"), ("on_the_network", "locked, on the network itself")):
        again = lock.get(key) or {}
        if again.get("status") == "scored":
            rows.append({"what": what, "name": "", "recorded": again["reward"],
                         "rescored": rewards.compute(again["parts"], reward_cfg)["reward"]})
    check = read_json(path / "heldout_check.json") if (path / "heldout_check.json").is_file() else {}
    for net, r in (check.get("networks") or {}).items():
        if r["status"] == "scored":
            rows.append({"what": f"locked on {net}", "name": "", "recorded": r["reward"],
                         "rescored": rewards.compute(r["parts"], reward_cfg)["reward"]})
    for row in rows:
        out(f"{row['what']:30s} {row['name'] or '':24s} recorded {row['recorded']:.6e}   {reward_cfg} -> {row['rescored']:.6e}")
    named = [r for r in rows if r["name"]]
    if named:
        top = max(named, key=lambda r: r["rescored"])
        out(f"highest under this reward: {top['what']} ({top['name']})")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--run-id")
    r.add_argument("--runs-dir")
    r.add_argument("--fake", help="a folder of scripted rounds: sets the backend to fake")
    r.add_argument("--sandbox", help="auto, hardened, container, landlock, seatbelt or process (default: the config's)")
    r.add_argument("--auth", choices=("login", "api_key"), help="who pays for the rounds (default: the config's)")
    r.add_argument("--budget-usd", type=float, help="api_key: what the whole run may spend, in dollars (default: the config's)")
    s = sub.add_parser("resume")
    s.add_argument("--run", required=True)
    s.add_argument("--fake")
    s.add_argument("--accept-changes", action="store_true")
    s.add_argument("--budget-usd", type=float, help="api_key: a new budget for the whole run, in dollars")
    f = sub.add_parser("finish")
    f.add_argument("--run", required=True)
    c = sub.add_parser("rescore")
    c.add_argument("--run", required=True)
    c.add_argument("--name", default="r_lambda", help="a reward of arcagent/rewards.py")
    c.add_argument("--lam", type=float)
    c.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="any other parameter of the reward")
    args = ap.parse_args(argv)
    from .backends import BackendError
    from .networks import ModelError
    from .sandbox import SandboxError

    try:
        if args.cmd == "run":
            over = {k: v for k, v in (("run_id", args.run_id), ("runs_dir", args.runs_dir), ("sandbox", args.sandbox),
                                      ("auth", args.auth), ("backend", args.backend), ("agent_model", args.agent_model),
                                      ("effort", args.effort), ("api_key_env", args.api_key_env)) if v}
            if args.budget_usd is not None:
                over["budget"] = {"run_usd": args.budget_usd}
            if args.fake:
                over["backend"] = "fake"
            start(run_config.load(args.config, over), fake_scripts=args.fake)
            return 0
        if args.cmd == "resume":
            return 0 if drive(Path(args.run), fake_scripts=args.fake, accept_changes=args.accept_changes,
                              budget_usd=args.budget_usd) == "finished" else 2
        if args.cmd == "finish":
            finish(Path(args.run))
            return 0
        rescore(Path(args.run), run_config.reward_from_args(args.name, args.param, args.lam))
        return 0
    except (run_config.ConfigError, hygiene.HygieneError, BackendError, ModelError, SandboxError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # what the user needs to read, without a traceback around it
        return 1


if __name__ == "__main__":
    sys.exit(main())
    r.add_argument("--backend", choices=("claude_code", "codex", "openai_api"), help="who answers the rounds (default: the config's); "
                   "with another backend give the run its own --run-id and --agent-model (docs/OPENAI.md)")
    r.add_argument("--agent-model", help="the model that answers (default: the config's)")
    r.add_argument("--effort", help="the model's reasoning effort (default: the config's)")
    r.add_argument("--api-key-env", help="api_key: the variable that holds the key (default: the config's)")
