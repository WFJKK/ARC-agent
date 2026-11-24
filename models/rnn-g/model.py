"""The network, written in the estimator language.

An input is n_ctx real numbers, x_0 to x_(n_ctx - 1). The network reads them one at a time, in that order.

The network is a recurrent network with one layer of d_hidden units, a ReLU and no biases. Its weights:

    input       d_hidden x 1
    recurrent   d_hidden x d_hidden
    readout     n_outputs x d_hidden

Its state h, a vector of d_hidden numbers, starts at zero. At each position t, in order, it becomes

    h = relu(input x_t + recurrent h)

where relu sets every negative number to zero. After the last position, readout h gives n_outputs numbers, the
logits, and the output is the position of the highest (the first one on a tie).

In an estimator, x is an expression of shape (n, n_ctx): n inputs, one row each (g.inputs(n) draws them from the
data set). logits(g, lang, x) is (n, n_outputs) and forward(g, lang, x) is (n,), the outputs. Every input costs
the same, so the network on a batch of n inputs costs n forward passes.
"""


def logits(g, lang, x):
    """The network's logits on a batch of inputs x, (n, n_ctx) -> (n, n_outputs)."""
    w, rec = g.weight("input")[:, 0], g.weight("recurrent")
    h = lang.maximum(x[:, 0][:, None] * w[None, :], 0.0)  # the first position: the state before it is zero
    for t in range(1, x.shape[1]):
        h = lang.maximum(x[:, t][:, None] * w[None, :] + h @ rec.T, 0.0)
    return h @ g.weight("readout").T


def forward(g, lang, x):
    """The network's outputs on a batch of inputs x, (n, n_ctx) -> (n,)."""
    return logits(g, lang, x).argmax(axis=1)
