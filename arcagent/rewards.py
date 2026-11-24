"""Reward functions: one number from what the verifier measured. A run config picks one by name.

The verifier runs an estimator many times, each run with fresh random draws, on the one network of the run, and
measures (verifier.py):

    accuracy   A          how often the network gives the task's answer, from inputs the verifier draws itself
    estimates  A_hat_k    the estimator's result in run k, for k = 1 .. runs
    cost       C_k        the operations of run k over the operations of one forward pass of the network

The parts kept with every receipt are their summaries: mse, the mean over runs of (A_hat_k - A)^2; cost, the mean
of C_k; and the mean, spread and range of the estimates. A finished run can be scored again under another reward
from them without running anything (python -m arcagent.loop rescore).

A new reward is a function here, taking the parts and its own parameters by name, with an entry in REWARDS and
its two texts for the agent's prompt in FORMULAS and EXPLANATIONS. This file is also placed in the agent's
sandbox, so nothing in it may say what a network's accuracy is.
"""

from __future__ import annotations

import inspect
import math


def r_lambda(parts: dict, lam: float = 1e-6, **_) -> dict:
    """-(A_hat - A)^2 - lambda * C, averaged over the runs: -mse - lambda * cost.

    The squared error and the compute are traded at the rate lambda: at lambda = 1e-6, an error of one percentage
    point (0.01^2 = 1e-4) weighs as much as 100 forward passes."""
    error, compute = -float(parts["mse"]), -float(lam) * float(parts["cost"])
    return {"reward": error + compute, "error_term": error, "cost_term": compute}


REWARDS = {"r_lambda": r_lambda}

# The reward as the agent's prompt states it, and a paragraph on how to read it. Parameters in braces are filled
# from the run config (or the function's defaults).
FORMULAS = {
    "r_lambda": "reward = mean over runs of [ -(estimate - A)^2 - lambda * cost ],   lambda = {lam:g}",
}
EXPLANATIONS = {
    "r_lambda": (
        "A is how often the network gives the task's answer, as the verifier measures it itself, and cost is\n"
        "a run's operations over the operations of one forward pass of the network. The\n"
        "reward is the negative mean squared error of the estimates, less lambda times the mean cost. Higher is\n"
        "better, and 0 is the most: the exact accuracy at no cost. An error of one percentage point, 0.01,\n"
        "costs 0.0001 of reward, as much as {per_point:g} forward passes: so {per_point:g} more forward passes that\n"
        "remove one point of error break even."),
}


def parameters(name: str) -> dict:
    """The parameters a reward takes, with their defaults: {name: default}."""
    sig = inspect.signature(REWARDS[name])
    return {k: p.default for k, p in sig.parameters.items() if k != "parts" and p.kind is not p.VAR_KEYWORD}


def check(cfg: dict) -> None:
    """Refuse a reward block that names no reward or a parameter the reward does not take (a typo would otherwise
    be ignored in silence)."""
    name = cfg.get("name", "r_lambda")
    if name not in REWARDS:
        raise ValueError(f"no reward named {name!r}; there are {sorted(REWARDS)}")
    unknown = sorted(set(cfg) - {"name"} - set(parameters(name)))
    if unknown:
        raise ValueError(f"the reward {name} does not take {unknown}; it takes {sorted(parameters(name))}")


def _filled(cfg: dict) -> tuple:
    check(cfg)
    name = cfg.get("name", "r_lambda")
    return name, {**parameters(name), **{k: v for k, v in cfg.items() if k != "name"}}


def lam_of(cfg: dict) -> float | None:
    """The reward's price of one forward pass, lambda, if it has one."""
    _, params = _filled(cfg)
    return float(params["lam"]) if "lam" in params else None


def warnings(cfg: dict) -> list:
    """What a reward block invites, in words, for the run's log."""
    name, params = _filled(cfg)
    out = []
    if "lam" in params:
        lam = float(params["lam"])
        if lam <= 0:
            out.append(f"lambda = {lam:g}: compute is free, so the best estimator runs the network on as many inputs "
                       "as the time limit allows")
        out.append("the reward scores only the estimate and its cost on this one network: a number measured on it "
                   "by sampling outside the estimator and written in as a constant scores near the top. The held-out "
                   "networks in the report are what shows it")
    return out


def formula(cfg: dict) -> str:
    """The reward as a formula, for the agent's prompt."""
    name, params = _filled(cfg)
    return FORMULAS[name].format(**params)


def explanation(cfg: dict) -> str:
    """How to read the formula, for the agent's prompt."""
    name, params = _filled(cfg)
    extra = {"per_point": 1e-4 / float(params["lam"])} if float(params.get("lam") or 0) > 0 else {"per_point": math.inf}
    return EXPLANATIONS[name].format(**{**params, **extra})


def compute(parts: dict, cfg: dict) -> dict:
    """cfg is the run config's reward block: {name, ...parameters}."""
    check(cfg)
    name = cfg.get("name", "r_lambda")
    params = {k: v for k, v in cfg.items() if k != "name"}
    out = REWARDS[name](parts, **params)
    return {"name": name, "params": params, **out}
