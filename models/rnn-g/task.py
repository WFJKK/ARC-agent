"""The task, written in the estimator language.

The task's answer for an input x, n_ctx real numbers, is the position of its second-largest number: the i for
which exactly one of the numbers is larger than x_i. Two numbers of an input are equal with probability 0.

answer(g, lang, x) gives it for a batch of inputs x, (n, n_ctx) -> (n,): a sort of each input, which costs
n_ctx * ceil(log2 n_ctx) per input.
"""


def answer(g, lang, x):
    """The task's answer on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return lang.argsort(x, axis=1)[:, -2]
