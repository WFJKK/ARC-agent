"""The task, written in the estimator language.

The task's answer for an input is the smallest of its tokens. An empty slot is not a token; it holds the value the
data set gives empty slots, which is above every token, so the smallest slot holds the smallest token.

answer(g, lang, x) gives it for a batch of inputs x, (n, n_ctx) -> (n,): n_ctx - 1 operations per input.
"""


def answer(g, lang, x):
    """The task's answer on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return x.min(axis=1)
