"""The reward R = -(A_hat - A)^2 - lambda C, averaged over runs, and the arithmetic of its write-up."""

import math

import pytest

from arcagent import rewards
from arcagent.verifier import sampling_benchmark


def parts(rms, cost):
    return {"mse": rms**2, "cost": cost}


def test_the_reward_is_minus_mse_minus_lambda_cost():
    r = rewards.compute(parts(0.02, 50.0), {"name": "r_lambda", "lam": 1e-6})
    assert math.isclose(r["reward"], -(0.02**2) - 1e-6 * 50)
    assert math.isclose(r["error_term"], -4e-4) and math.isclose(r["cost_term"], -5e-5)
    assert rewards.compute({"mse": 0.0, "cost": 0.0}, {"name": "r_lambda"})["reward"] == 0  # the most there is


def test_the_worked_example_of_round_five():
    """The write-up's numbers: root mean squared error 1.286 percentage points (0.0128565), mean cost 109.86
    forward passes, lambda 1e-6."""
    r = rewards.compute(parts(0.0128565, 109.86), {"name": "r_lambda", "lam": 1e-6})["reward"]
    assert round(r, 8) == -0.00027515
    # one percentage point of error weighs as much as 100 forward passes: spending 100 to remove it breaks even
    assert math.isclose(0.01**2, 1e-6 * 100)
    # the constant guess 95% against A = 0.953006 misses by 0.3006 points, costs nothing, and scores higher
    const = rewards.compute({"mse": (0.95 - 0.953006) ** 2, "cost": 0.0}, {"name": "r_lambda", "lam": 1e-6})["reward"]
    assert const > r and math.isclose(const, -9.036036e-06, rel_tol=1e-6)
    # sampling had 2.031 points at the same cost: 59.9% lower mean squared error for the estimator
    assert round(1 - (1.286 / 2.031) ** 2, 3) == 0.599


def test_the_sampling_benchmark():
    A = 0.953006
    b = sampling_benchmark(A, mse=0.0128565**2, cost=109.86, sample_cost=1.0, lam=1e-6)
    var = A * (1 - A)
    assert math.isclose(b["samples_at_cost"], 109.86) and math.isclose(b["mse_at_cost"], var / 109.86)
    assert math.isclose(b["mse_reduction"], 1 - 0.0128565**2 / (var / 109.86))
    assert math.isclose(b["best_samples"], math.sqrt(var / 1e-6)) and math.isclose(b["best_reward"], -2 * math.sqrt(1e-6 * var))
    # with this repository's price of a sample on the challenge network (1.0177 forward passes)
    b = sampling_benchmark(A, mse=0.0128565**2, cost=109.86, sample_cost=2989 / 2937, lam=1e-6)
    assert 2.03 < 100 * b["rms_error_at_cost"] < 2.04
    # below one sample, sampling has nothing to say
    assert sampling_benchmark(A, mse=0.0, cost=0.5, sample_cost=1.0, lam=1e-6)["mse_reduction"] is None


def test_a_reward_block_is_checked():
    with pytest.raises(ValueError, match="no reward named"):
        rewards.check({"name": "r_alpha"})
    with pytest.raises(ValueError, match="does not take"):
        rewards.check({"name": "r_lambda", "lambda": 1e-6})
    assert rewards.parameters("r_lambda") == {"lam": 1e-6}
    assert rewards.lam_of({"name": "r_lambda", "lam": 2e-6}) == 2e-6


def test_the_texts_for_the_prompt():
    cfg = {"name": "r_lambda", "lam": 1e-6}
    assert "lambda = 1e-06" in rewards.formula(cfg)
    assert "as much as 100 forward passes" in rewards.explanation(cfg)
    assert "as much as 1000 forward passes" in rewards.explanation({"name": "r_lambda", "lam": 1e-7})
    w = rewards.warnings(cfg)
    assert len(w) == 1 and "written in as a constant" in w[0]
    assert rewards.warnings(cfg, on_copies=True) == []  # on copies of the network that number does not win
    assert any("compute is free" in x for x in rewards.warnings({"name": "r_lambda", "lam": 0}))
