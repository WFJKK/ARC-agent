"""Model folders: a model's networks, the network and the task written in the estimator language, and what the
agent is given.

A model folder (models/<name>/) holds

    config.yaml   sizes, the input space, the data set and the task (each with the sentence the agent is told),
                  the weights' names and shapes, the networks (id, file, sha256)
    model.py      logits(g, lang, x) and forward(g, lang, x): the network on a batch of inputs, in the language
    task.py       answer(g, lang, x): the task's answer on a batch of inputs, in the language
    zoo/*.npz     the networks' weights (no pickle)

The agent is told the task: an accuracy means nothing without it. What it is not told is in withheld/<name>/
(task.yaml: where the networks come from, how often each gives the task's answer, the strings the hygiene check
looks for; references.py: reference estimators for the report), which only the verifier's side reads.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from . import lang
from .util import REPO_ROOT, read_yaml, sha256_file


class ModelError(RuntimeError):
    pass


def _load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        if not (self.path / "config.yaml").is_file():
            raise ModelError(f"{self.path} is not a model folder (no config.yaml)")
        self.config = read_yaml(self.path / "config.yaml")
        self.name = self.config["name"]
        self.sizes = dict(self.config["sizes"])
        self.n_ctx = int(self.sizes["n_ctx"])
        self.data_set = dict(self.config.get("data_set") or {"kind": "all"})
        try:
            kind = lang.data_set_kind(self.data_set)
        except ValueError as exc:
            raise ModelError(f"the data set of {self.name} is not well described: {exc}") from None
        self.real = kind == "normal"
        space = list(self.config["input_space"])
        if self.real != all(n == "real" for n in space) or (not self.real and not all(isinstance(n, int) for n in space)):
            raise ModelError("input_space says `real` for every position of a data set of numbers (kind normal), "
                             "and a whole number of tokens for every position otherwise")
        self.input_space = space if self.real else [int(n) for n in space]
        if len(self.input_space) != self.n_ctx:
            raise ModelError("input_space needs one entry per position")
        if not self.data_set.get("says") or not (self.config.get("task") or {}).get("says"):
            raise ModelError("a model needs the sentences the agent is told about its data set and its task "
                             "(data_set.says, task.says)")
        self._module = None
        self._task = None

    # what the agent is told in words
    @staticmethod
    def _words(text) -> str:
        return " ".join(str(text).split())

    def data_set_text(self) -> str:
        return self._words(self.data_set["says"])

    def task_text(self) -> str:
        return self._words(self.config["task"]["says"])

    # the networks
    def network_ids(self) -> list:
        return [n["id"] for n in self.config["networks"]]

    def network(self, network_id: str) -> dict:
        for n in self.config["networks"]:
            if n["id"] == network_id:
                return n
        raise ModelError(f"{network_id} is not a network of {self.name}; it has {self.network_ids()}")

    def weights_path(self, network_id: str) -> Path:
        return self.path / self.network(network_id)["weights"]

    def load_weights(self, network_id: str) -> dict:
        entry = self.network(network_id)
        path = self.path / entry["weights"]
        if sha256_file(path) != entry["sha256"]:
            raise ModelError(f"the weights of {network_id} do not match the hash in config.yaml")
        with np.load(path, allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files}
        shapes = {k: list(v.shape) for k, v in weights.items()}
        if shapes != {k: list(v) for k, v in self.config["weights"].items()}:
            raise ModelError(f"the weights of {network_id} have shapes {shapes}, not those of config.yaml")
        return weights

    # the network and the task, in the language
    def source(self) -> str:
        return (self.path / "model.py").read_text()

    def task_source(self) -> str:
        return (self.path / "task.py").read_text()

    def module(self):
        if self._module is None:
            self._module = _load_py(self.path / "model.py", f"arcagent_model_{self.name.replace('-', '_')}")
        return self._module

    def task_module(self):
        if self._task is None:
            self._task = _load_py(self.path / "task.py", f"arcagent_task_{self.name.replace('-', '_')}")
        return self._task

    def graph(self, weights: dict, seed: int = 0) -> lang.Graph:
        return lang.Graph(weights, self.data_set, self.input_space, seed)

    def _batch_ops(self, weights: dict, n: int, what: str) -> int:
        g = self.graph(weights)
        x = g.const(np.zeros((n, self.n_ctx), dtype=np.float64 if self.real else np.int64))
        out = self.module().forward(g, lang, x) if what == "forward" else self.task_module().answer(g, lang, x)
        if out.shape != (n,):
            raise ModelError(f"{what} gives shape {out.shape} on {n} inputs, not ({n},)")
        return lang.Evaluator(g.entry(out), weights).run([0])["ops"]

    def forward_ops(self, weights: dict) -> int:
        """F: the operations of one forward pass. The network must cost the same on every input, so a batch of n
        costs n F; this is checked on batches of 1, 2 and 5."""
        one = self._batch_ops(weights, 1, "forward")
        if self._batch_ops(weights, 2, "forward") != 2 * one or self._batch_ops(weights, 5, "forward") != 5 * one:
            raise ModelError("the network's cost on a batch is not the batch size times its cost on one input")
        return one

    def answer_ops(self, weights: dict) -> int:
        """The operations of the task's answer for one input (the same for every input)."""
        one = self._batch_ops(weights, 1, "answer")
        if self._batch_ops(weights, 3, "answer") != 3 * one:
            raise ModelError("the task's cost on a batch is not the batch size times its cost on one input")
        return one

    def monte_carlo(self, weights: dict, n: int) -> dict:
        """Plain sampling as an estimator: n inputs drawn afresh in every run, the share on which the network gives
        the task's answer. It costs n times (the draws, a forward pass, the answer and one comparison) plus the
        sum and the division."""
        g = self.graph(weights)
        x = g.inputs(int(n))
        right = self.module().forward(g, lang, x) == self.task_module().answer(g, lang, x)
        return g.entry(right.sum() / float(n))

    def sample_ops(self, weights: dict) -> int:
        """What one more input costs plain sampling: the operations of monte_carlo(n + 1) less those of monte_carlo(n)."""
        a = lang.Evaluator(self.monte_carlo(weights, 1), weights).run([0])["ops"]
        b = lang.Evaluator(self.monte_carlo(weights, 2), weights).run([0])["ops"]
        return b - a

    def accuracy(self, weights: dict, n: int, seed: int, chunk: int = 32768) -> dict:
        """How often the network gives the task's answer, measured by running it on n inputs drawn from the data set
        with the seed, chunk inputs at a time: chunk k is monte_carlo(chunk) run with the seed [seed, k]. The network
        and the task are run through the interpreter, as an estimator would run them."""
        n, chunk = int(n), int(chunk)
        if n < 1 or chunk < 1 or n % chunk:
            raise ValueError("the inputs must be a positive multiple of the chunk")
        runs = n // chunk
        r = lang.run_estimator(self.monte_carlo(weights, chunk), weights, runs, int(seed))
        correct = int(sum(int(round(e * chunk)) for e in r["estimates"]))
        a = correct / n
        return {"accuracy": a, "correct": correct, "n": n, "seed": int(seed), "chunk": chunk,
                "standard_error": math.sqrt(a * (1 - a) / n)}

    # what the agent is given about the network besides its weights, model.py and task.py
    def info(self) -> dict:
        return {"sizes": self.sizes, "input_space": self.input_space, "n_outputs": int(self.config["n_outputs"]),
                "weights": {k: list(v) for k, v in self.config["weights"].items()},
                "data_set": {k: v for k, v in self.data_set.items() if k != "says"}}

    # the verifier's side only
    def withheld(self) -> Path:
        return self.path.parent.parent / "withheld" / self.name

    def task_record(self) -> dict:
        p = self.withheld() / "task.yaml"
        return read_yaml(p) if p.is_file() else {}

    def hygiene(self) -> list:
        return [str(s) for s in (self.task_record().get("hygiene") or [])]

    def reference_entries(self, weights: dict) -> dict:
        """The withheld reference estimators, built for these weights: {name: estimator}. Empty without them."""
        p = self.withheld() / "references.py"
        if not p.is_file():
            return {}
        mod = _load_py(p, f"arcagent_withheld_{self.name.replace('-', '_')}")
        out = {}
        for name, fn in getattr(mod, "ESTIMATORS", {}).items():
            g = self.graph(weights)
            out[name] = g.entry(fn(g, lang, self.module(), self.task_module()))
        return out


def load_model(path) -> Model:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return Model(p)
