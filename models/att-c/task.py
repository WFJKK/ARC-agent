"""The task, written in the estimator language.

The task's answer for an input is the largest of its tokens. An empty slot is not a token, and the answer leaves it
out: it holds the value the data set gives empty slots, which is above every token.

answer(g, lang, x) gives it for a batch of inputs x, (n, n_ctx) -> (n,): per slot a comparison and a where, then
the largest, 3 n_ctx - 1 operations per input.
"""


def answer(g, lang, x):
    """The task's answer on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return lang.where(x == g.data_set["empty"], -1, x).max(axis=1)
