"""One agent session: its two tools, their budgets and their log.

A session lives in a round's folder:

    round_k/session/
        session.json          what the tools need (model, network, limits, reward) and what was used so far
        tools.jsonl           every tool call, appended as it happens: tool, arguments, outcome, seconds
        python/<n>.py, .txt   every program run_python was given, in full, and what went back to the agent
        python/<n>.audit.json what the guard inside Python refused that program, when it refused anything or its
                              record is incomplete
        workspace/estimators/ the estimators saved from run_python (and the ones carried from earlier rounds)
        submissions/<n>/      one folder per submit_estimator call: estimator.json, receipt.json, runs.npz

The tools:

- run_python(code): run code in the sandbox with the network's weights, the estimator language, the network's and
  the task's definitions in scope; returns what it printed. save_estimator(name, estimator) in that code keeps an
  estimator in the session's workspace. Counts against the python budget.
- submit_estimator(name): hand a saved estimator to the verifier, which runs it on its own seeds and returns what
  the agent may see of the receipt (verifier.agent_view). Counts against the submission budget.

The verifier is the judge; this file is its client. Budgets persist in session.json, so a session that is started
again (a resumed run) continues them, and its earlier submissions stay.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import lang
from .networks import load_model
from .util import append_jsonl, atomic_write, now, read_json, read_jsonl, write_json

MAX_TEXT = 20000  # characters of program output returned to the agent
MAX_SOURCE = 200_000  # characters of one program
NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
REFERENCES = "references.jsonl"  # in the run folder: the verifier's measurements of the networks' accuracies


class ToolError(ValueError):
    """A tool refused the call: the message goes back to the agent."""


WRAPPER = '''"""run_python: the agent's code, with the network, the task and the estimator language in scope."""
import contextlib, io, traceback, warnings
import numpy as np

# On some machines numpy's matrix product raises floating-point flags when nothing went wrong, and prints
# "divide by zero / overflow / invalid value encountered in matmul" though the result is exact. Elsewhere
# the same code prints nothing. Those three messages are not shown, so the output is the same on every machine;
# a real overflow still shows in the values (inf, nan).
warnings.filterwarnings("ignore", message=r"(divide by zero|overflow|invalid value) encountered in matmul",
                        category=RuntimeWarning)

import lang
import model
import rewards
import task

CODE = @@CODE@@


def make_result(weights, info, extra):
    estimators = dict(extra["estimators"])
    saved = {}
    F = info["forward_ops"]

    def graph(seed=0):
        """A new lang.Graph on these weights and the data set. Its example run uses the seed."""
        return lang.Graph(weights, info["data_set"], info["input_space"], seed)

    def run(estimator, runs=1, seed=0):
        """The estimator run `runs` times as the verifier runs it, run k with the seed [seed, k]: the estimates,
        the operations and the random numbers drawn in each run, and the cost of each in forward passes."""
        lang.check_structure(estimator, extra["limits"]["max_nodes"], extra["limits"]["max_const_scalars"])
        r = lang.run_estimator(estimator, weights, runs, seed)
        return {**r, "cost": r["ops"] / F}

    def score_against(estimator, accuracy, runs=256, seed=0):
        """The reward the verifier would give the estimator if the network's accuracy were `accuracy`, over `runs`
        runs. The verifier measures the accuracy itself; this one is yours."""
        r = run(estimator, runs, seed)
        parts = {"mse": float(np.mean((r["estimates"] - float(accuracy)) ** 2)), "cost": float(r["cost"].mean())}
        return {**parts, "estimate_mean": float(r["estimates"].mean()), "estimate_std": float(r["estimates"].std()),
                "reward": rewards.compute(parts, extra["reward"])["reward"]}

    def sample(n, seed=0):
        """n inputs drawn from the data set with numpy.random.default_rng(seed), one row each."""
        return lang.draw_inputs(info["data_set"], info["input_space"], int(n), np.random.default_rng(seed))

    def outputs(x, chunk=32768):
        """The network's outputs and the task's answers on the inputs x (a numpy array, one row each), computed
        with model.forward and task.answer through the interpreter."""
        x = np.asarray(x)
        got, want = [], []
        for i in range(0, len(x), chunk):
            g = graph()
            xs = g.const(x[i:i + chunk])
            both = lang.stack([model.forward(g, lang, xs), task.answer(g, lang, xs)], axis=0)
            v = lang.Evaluator(g.entry(both), weights).run([0])["estimate"]
            got.append(v[0])
            want.append(v[1])
        return np.concatenate(got), np.concatenate(want)

    def save_estimator(name, estimator):
        """Keep an estimator in the session's workspace under a name, to submit it or to build on it later."""
        size = lang.check_structure(estimator, extra["limits"]["max_nodes"], extra["limits"]["max_const_scalars"])
        saved[str(name)] = estimator
        estimators[str(name)] = estimator
        return size

    scope = {"weights": weights, "info": info, "np": np, "lang": lang, "model": model, "task": task, "graph": graph,
             "run": run, "score_against": score_against, "sample": sample, "outputs": outputs,
             "estimators": estimators, "save_estimator": save_estimator, "__name__": "__main__"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(CODE, "<run_python>", "exec"), scope)
        except BaseException:
            buf.write(traceback.format_exc())
    return {"stdout": buf.getvalue()[-20000:], "estimators": saved}
'''

TOOLS = {
    "run_python": {
        "description": (
            "Run Python code in a sandbox and get back what it printed. In scope: `weights` (a dict of numpy arrays), "
            "`info` (sizes, the input space, the data set, the operations of one forward pass, the verifier's runs and "
            "the limits), `np`, `lang` (the estimator language), `model` (the network's definition: model.logits and "
            "model.forward), `task` (the task's definition: task.answer), `graph(seed=0)` (a new lang.Graph on these "
            "weights), `run(estimator, runs=1, seed=0)` (its estimates and costs, as the verifier computes them), "
            "`score_against(estimator, accuracy, runs=256, seed=0)` (the reward if the network's accuracy were the "
            "number you give), `sample(n, seed=0)` (inputs drawn from the data set with numpy), `outputs(x)` (the "
            "network's outputs and the task's answers on inputs x), `estimators` (the saved estimators by name) and "
            "`save_estimator(name, estimator)`. The code may read its own folder and Python's libraries and write "
            "only to tmp/ in its folder; other files, the network and other programs are closed to it "
            "(PermissionError). Nothing persists between calls except saved estimators. Each call counts against the "
            "python budget."),
        "schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    "submit_estimator": {
        "description": (
            "Hand a saved estimator to the verifier. It runs the estimator many times on seeds of its own and returns "
            "the result: status scored or refused (with the reason), the reward, the mean squared error of the "
            "estimates, their mean and spread, and the cost in forward passes (mean, least and most). Each call counts "
            "against the submission budget."),
        "schema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the name the estimator was saved under"}}, "required": ["name"]},
    },
}


def verifier_settings(cfg: dict) -> dict:
    from .verifier import DEFAULTS, ENTRY_DEFAULTS

    return {**DEFAULTS, **(cfg.get("verifier") or {})}, {**ENTRY_DEFAULTS, **(cfg.get("entry") or {})}


def agent_info(model, forward_ops: int, cfg: dict) -> dict:
    """What the agent's code is handed about the network, its data set and the verifier. No name in it says more
    than the model folder's own agent-facing files."""
    ver, limits = verifier_settings(cfg)
    return {**model.info(), "forward_ops": int(forward_ops), "runs": int(ver["runs"]),
            "reference_inputs": int(ver["reference_inputs"]),
            "limits": {"max_nodes": int(limits["max_nodes"]), "max_const_scalars": int(limits["max_const_scalars"])}}


def box_files(model) -> dict:
    """The modules the agent's code may import, as they are placed in the sandbox folder."""
    here = Path(__file__).parent
    return {"lang.py": (here / "lang.py").read_text(), "rewards.py": (here / "rewards.py").read_text(),
            "model.py": model.source(), "task.py": model.task_source()}


def surface(model, forward_ops: int, cfg: dict) -> dict:
    """Every text of a session the agent can reach, for the hygiene check: the tools' descriptions, the files of the
    sandbox folder, the wrapper and what its code is handed. (The prompt is checked with them.)"""
    out = {f"tool {name}": json.dumps(spec) for name, spec in TOOLS.items()}
    out.update({f"sandbox file {name}": text for name, text in box_files(model).items()})
    out["sandbox file harness.py"] = (Path(__file__).parent / "harness.py").read_text()
    out["sandbox wrapper"] = WRAPPER
    out["sandbox info"] = json.dumps(agent_info(model, forward_ops, cfg))
    out["weights' names"] = " ".join(model.config["weights"])
    return out


def cached_references(run_path: Path) -> dict:
    """{network: the verifier's measurement of its accuracy}, as the run folder keeps them."""
    out = {}
    for line in read_jsonl(Path(run_path) / REFERENCES):
        out[line["network"]] = line
    return out


class Session:
    TOOLS = TOOLS

    def __init__(self, path: Path):
        self.path = Path(path)
        self.state_path = self.path / "session.json"
        self.state = read_json(self.state_path)
        self.saved_dir = self.path / "workspace" / "estimators"
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self._verifier = None
        self._model = None

    # creation --------------------------------------------------------------------------------
    @classmethod
    def create(cls, path: Path, cfg: dict, network_id: str, run_path: Path, carried: dict | None = None) -> "Session":
        """Make (or reopen) a round's session folder. Budgets carry over if it exists. `carried` are estimators of
        earlier rounds, {name: estimator}, placed in the workspace."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "session.json"
        if not state_path.exists():
            s = cfg["session"]
            state = {
                "model": str(cfg["model"]), "network": network_id, "run_path": str(Path(run_path).resolve()),
                "budgets": {"run_python": int(s["python_calls"]), "submit_estimator": int(s["submissions"])},
                "used": {"run_python": 0, "submit_estimator": 0},
                "python": dict(cfg["python"]), "verifier": dict(cfg["verifier"]), "entry": dict(cfg["entry"]),
                "reward": dict(cfg["reward"]), "sandbox": cfg["sandbox"], "sandbox_image": cfg.get("sandbox_image"),
                "sandbox_dir": cfg.get("sandbox_dir"), "runs_dir": cfg.get("runs_dir"), "created": now(),
            }
            write_json(state_path, state)
            session = cls(path)
            for name, est in (carried or {}).items():
                write_json(session.saved_dir / f"{name}.json", est)
            return session
        return cls(path)

    def reload(self) -> None:
        """Re-read the saved state: the tools run in the tool server's process, which saves its counters here."""
        self.state = read_json(self.state_path)

    # shared ----------------------------------------------------------------------------------
    def _log(self, tool: str, args: dict, outcome: dict, seconds: float) -> None:
        short = {k: (v if not isinstance(v, str) or len(v) < 4000 else v[:4000] + f"... ({len(v)} chars)") for k, v in args.items()}
        append_jsonl(self.path / "tools.jsonl", {"at": now(), "tool": tool, "args": short, "outcome": outcome, "seconds": round(seconds, 2)})

    def _spend(self, tool: str) -> None:
        used, budget = self.state["used"][tool], self.state["budgets"][tool]
        if used >= budget:
            raise ToolError(f"the {tool} budget of this session ({budget} calls) is used up")
        self.state["used"][tool] = used + 1
        write_json(self.state_path, self.state)

    def remaining(self) -> dict:
        return {t: self.state["budgets"][t] - self.state["used"][t] for t in self.state["budgets"]}

    def model(self):
        if self._model is None:
            self._model = load_model(self.state["model"])
        return self._model

    def cfg(self) -> dict:
        return {"verifier": self.state["verifier"], "entry": self.state["entry"], "reward": self.state["reward"]}

    def verifier(self):
        if self._verifier is None:
            from .verifier import Verifier

            ref = cached_references(Path(self.state["run_path"])).get(self.state["network"])
            v = Verifier(self.model(), self.state["network"], self.cfg())
            if ref is not None:
                try:
                    v.set_reference(ref)
                except ValueError:
                    pass  # made under other settings: the verifier measures it again
            self._verifier = v
        return self._verifier

    def estimators(self) -> dict:
        out = {}
        for p in sorted(self.saved_dir.glob("*.json")):
            if NAME.match(p.stem) and p.is_file() and not p.is_symlink():
                try:
                    out[p.stem] = read_json(p)
                except (OSError, ValueError):
                    pass
        return out

    def info(self) -> dict:
        return agent_info(self.model(), self.verifier().F, self.cfg())

    def box_files(self) -> dict:
        return box_files(self.model())

    # the tools -------------------------------------------------------------------------------
    def run_python(self, code: str) -> dict:
        t0 = time.monotonic()
        args = {"code": code}
        try:
            if not isinstance(code, str) or len(code) > MAX_SOURCE:
                raise ToolError("code must be a string of at most 200,000 characters")
            self._spend("run_python")
            n = self.state["used"]["run_python"]
            atomic_write(self.path / "python" / f"{n:03d}.py", code)  # in full: the log line below is cut short
            from .runner import Limits, run_recipe
            from .sandbox import choose, protected_paths

            m = self.model()
            limits = Limits.from_config(self.state["python"])
            cfg = {"sandbox": self.state["sandbox"], "sandbox_image": self.state["sandbox_image"], "runs_dir": self.state["runs_dir"]}
            sb = choose(cfg, protected_paths(cfg, m, Path(self.state["run_path"])))
            extra = {"estimators": self.estimators(), "reward": self.state["reward"],
                     "limits": {k: int(self.state["entry"][k]) for k in ("max_nodes", "max_const_scalars")}}
            ex = run_recipe(WRAPPER.replace("@@CODE@@", repr(code)), self.box_files(), m.load_weights(self.state["network"]),
                            self.info(), extra, limits, self.state["sandbox_dir"], sandbox=sb)
            saved, problems = [], []
            if ex.status == "ok" and isinstance(ex.result, dict):
                output = str(ex.result.get("stdout", ""))
                got = ex.result.get("estimators")
                for name, est in (got.items() if isinstance(got, dict) else []):
                    try:
                        if not NAME.match(str(name)):
                            raise lang.EntryError("an estimator's name is 1 to 40 letters, digits, - or _")
                        lang.check_structure(est, self.state["entry"]["max_nodes"], self.state["entry"]["max_const_scalars"])
                        write_json(self.saved_dir / f"{name}.json", est)
                        saved.append(str(name))
                    except lang.EntryError as exc:
                        problems.append(f"{name}: not saved: {exc}")
            else:
                output = f"({ex.status}) {ex.error or ''}"
            out = {"status": ex.status, "output": output[-MAX_TEXT:], "saved_estimators": sorted(saved), "runtime_s": ex.runtime_s,
                   "sandbox": ex.mode, "remaining": self.remaining()}
            if problems:
                out["problems"] = problems
            atomic_write(self.path / "python" / f"{n:03d}.txt", out["output"])
            audit = ex.audit or {"present": False, "complete": False, "n_blocked": 0, "blocked": []}
            record = "complete" if audit.get("complete") else ("incomplete" if audit.get("present") else "missing")
            if audit.get("n_blocked") or record != "complete":
                write_json(self.path / "python" / f"{n:03d}.audit.json", {**audit, "record": record, "status": ex.status})
            self._log("run_python", args, {"status": ex.status, "output": output[:500], "saved": sorted(saved), "sandbox": ex.mode,
                                           "blocked": int(audit.get("n_blocked") or 0), "guard_record": record}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("run_python", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def submit_estimator(self, name: str) -> dict:
        from .verifier import agent_view

        t0 = time.monotonic()
        args = {"name": name}
        try:
            if not isinstance(name, str) or not NAME.match(name):
                raise ToolError("name must be the name an estimator was saved under")
            path = self.saved_dir / f"{name}.json"
            if not path.is_file() or path.is_symlink():
                raise ToolError(f"no saved estimator named {name!r}; the saved estimators are {sorted(self.estimators())}")
            est = read_json(path)
            self._spend("submit_estimator")
            n = self.state["used"]["submit_estimator"]
            folder = self.path / "submissions" / f"{n:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "estimator.json", est)
            receipt = self.verifier().score(est, keep_runs=True)
            runs = receipt.pop("runs_data", None)
            if runs is not None:
                np.savez(folder / "runs.npz", **{k: np.asarray(v) for k, v in runs.items()})
            receipt.update(name=name, submission=n, at=now())
            write_json(folder / "receipt.json", receipt)
            self._log("submit_estimator", args, {"status": receipt["status"], "reward": receipt["reward"], "reason": receipt["reason"]},
                      time.monotonic() - t0)
            return {**agent_view(receipt), "remaining": self.remaining()}
        except ToolError as exc:
            self._log("submit_estimator", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def receipts(self) -> list:
        """The receipts of this session's submissions, in order."""
        out = []
        sub = self.path / "submissions"
        if sub.is_dir():
            for d in sorted(p for p in sub.iterdir() if p.is_dir() and p.name.isdigit()):
                try:
                    out.append(read_json(d / "receipt.json"))
                except (OSError, ValueError):
                    pass  # a submission cut off before its receipt: it spent its budget and scored nothing
        return out

    def call(self, tool: str, arguments: dict) -> dict:
        if tool not in self.TOOLS:
            raise ToolError(f"no tool {tool!r}; the tools are {sorted(self.TOOLS)}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        schema = self.TOOLS[tool]["schema"]
        missing = [k for k in schema["required"] if k not in arguments]
        if missing:
            raise ToolError(f"{tool} needs {missing}")
        extra = sorted(set(arguments) - set(schema["properties"]))
        if extra:
            raise ToolError(f"{tool} does not take {extra}")
        return getattr(self, tool)(**arguments)
