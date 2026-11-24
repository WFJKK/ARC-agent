"""The network, written in the estimator language.

An input is n_ctx slots, each holding a token (a whole number) or the empty value; the tokens fill the first slots.
The network is a transformer with one layer and one attention head, attention only (no MLP, no layer norm, no
biases), with learned positions, read at the slot of the last token given. Its weights:

    embed     (number of token values) x d_model   the empty value's row is all zeros
    pos       n_ctx x d_model
    query, key, value, out                        d_model x d_model
    unembed   d_model x n_outputs

For an input x with k tokens (k >= 1):

    v_i = embed[x_i] + pos[i]                     one vector per slot
    r = v_(k-1)                                   the slot of the last token given
    q = r query / sqrt(d_model)
    s_i = (v_i key) . q   for the k slots that hold a token; an empty slot takes no part
    a = softmax(s)                                over those slots
    logits = (r + (sum_i a_i v_i value) out) unembed

and the output is the position of the highest logit (the first one on a tie): logit j stands for token j.

In an estimator, x is an expression of shape (n, n_ctx): n inputs, one row each (g.inputs(n) draws them from the
data set). logits(g, lang, x) is (n, n_outputs) and forward(g, lang, x) is (n,), the outputs. Every input costs
the same, so the network on a batch of n inputs costs n forward passes. The slot of the last token is read through
a flat index into the batch's vectors, at the price of reading one vector and one addition.
"""

import math


def logits(g, lang, x):
    """The network's logits on a batch of inputs x, (n, n_ctx) -> (n, n_outputs)."""
    embed, pos = g.weight("embed"), g.weight("pos")
    n_ctx, d = pos.shape
    n, empty = x.shape[0], embed.shape[0] - 1
    v = lang.take(embed, x, axis=0) + pos                       # (n, n_ctx, d): a vector per slot
    k = (x != empty).sum(axis=1)                                 # (n,): the number of tokens given
    # the vector of slot k - 1 of input b is row b * n_ctx + k of `rows`, whose first row is never read
    rows = lang.concat([pos[:1], v.reshape(n * n_ctx, d)], axis=0)
    first = g.arange(n * n_ctx).reshape(n, n_ctx)[:, 0]         # b * n_ctx, free
    r = lang.take(rows, first + k, axis=0)                       # (n, d)
    q = (r @ g.weight("query")) * (1 / math.sqrt(g.weight("query").shape[1]))
    s = ((v @ g.weight("key")) @ q.reshape(n, d, 1)).reshape(n, n_ctx)
    s = lang.where(g.arange(n_ctx)[None, :] < k[:, None], s, -1e30)   # empty slots take no part
    head = (lang.softmax(s, axis=1).reshape(n, 1, n_ctx) @ (v @ g.weight("value"))).reshape(n, d)
    return (r + head @ g.weight("out")) @ g.weight("unembed")


def forward(g, lang, x):
    """The network's outputs on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return logits(g, lang, x).argmax(axis=1)
