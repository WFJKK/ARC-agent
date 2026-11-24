"""The task, written in the estimator language.

The task's answer for an input is the largest of its tokens.

answer(g, lang, x) gives it for a batch of inputs x, (n, n_ctx) -> (n,): n_ctx - 1 operations per input.
"""


def answer(g, lang, x):
    """The task's answer on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return x.max(axis=1)
