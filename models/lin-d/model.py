"""The network, written in the estimator language.

An input is n_ctx tokens, whole numbers; in a data set of varying lengths, a slot without a token holds the empty
value, whose embedding column is all zeros, so that it adds nothing. The network adds up one embedding column per
slot and maps the sum to logits with two matrices. Its weights:

    embed     d_model x (number of token values)
    hidden    d_model x d_model
    unembed   n_outputs x d_model

so that, for an input x,

    h = embed[:, x_0] + ... + embed[:, x_(n_ctx - 1)]
    logits = unembed (hidden h)

and the output is the position of the highest logit (the first one on a tie): logit j stands for token j.

In an estimator, x is an expression of shape (n, n_ctx): n inputs, one row each (g.inputs(n) draws them from the
data set). logits(g, lang, x) is (n, n_outputs) and forward(g, lang, x) is (n,), the outputs. Every input costs
the same, so the network on a batch of n inputs costs n forward passes. lang.take_sum adds up the columns of each
input at d_model operations per column, the price of a matrix applied to a sum of one-hot vectors.
"""


def logits(g, lang, x):
    """The network's logits on a batch of inputs x, (n, n_ctx) -> (n, n_outputs)."""
    h = lang.take_sum(g.weight("embed"), x, axis=1)  # (d_model, n): each input's columns added up
    return (g.weight("unembed") @ (g.weight("hidden") @ h)).T


def forward(g, lang, x):
    """The network's outputs on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return logits(g, lang, x).argmax(axis=1)
