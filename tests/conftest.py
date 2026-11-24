import numpy as np
import pytest


@pytest.fixture()
def toy():
    """Small weights for tests of the language itself."""
    rng = np.random.default_rng(0)
    return {"A": rng.normal(size=(3, 4)), "B": rng.normal(size=(4, 5)), "v": rng.normal(size=4),
            "T": rng.normal(size=(6, 6)), "K": rng.integers(0, 6, size=(6, 6))}


@pytest.fixture(scope="session")
def model():
    """The small model: a forward pass is 78 operations, so whole runs take seconds."""
    from arcagent.networks import load_model

    return load_model("models/rnn-b")
