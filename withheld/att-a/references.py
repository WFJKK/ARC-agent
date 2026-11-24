"""Reference estimators for att-a, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g, the estimator language lang, and the model's and the task's modules, and returns the
estimate as an expression.

An input is one token a, or two tokens a and b. Everything the network computes splits into tables over single
tokens, which the estimator below builds once and combines for every input:

    v0[a] = embed[a] + pos[0],  v1[b] = embed[b] + pos[1]          a token in the first slot, in the second
    one token:   logits = v0[a] unembed + v0[a] (value out unembed)             (it attends to itself alone)
    two tokens:  read at the second slot, q = v1[b] query / sqrt(d)
                 s0 = (v0[a] key) . q,  s1 = (v1[b] key) . q,  (w0, w1) = softmax(s0, s1)
                 logits = v1[b] unembed + w0 v0[a] (value out unembed) + w1 v1[b] (value out unembed)
"""

import math


def every_input(g, lang, model, task):
    """The accuracy itself: every input of one token and of two, through the tables above. Half the data set is
    one token, half is two."""
    embed, pos = g.weight("embed"), g.weight("pos")
    unembed = g.weight("unembed")
    n = embed.shape[0] - 1                       # token values; the last row is the empty slot's
    d = pos.shape[1]
    tokens = g.arange(n)
    v0, v1 = embed[:n] + pos[0], embed[:n] + pos[1]
    through = (g.weight("value") @ g.weight("out")) @ unembed       # what the head's output adds to the logits
    t0, t1 = v0 @ through, v1 @ through                             # (n, outputs) each
    # one token: the task's answer is the token
    one = ((v0 @ unembed + t0).argmax(axis=1) == tokens).mean()
    # two tokens, a in the first slot (rows) and b in the second (columns)
    q = (v1 @ g.weight("query")) * (1 / math.sqrt(d))
    s0 = (v0 @ g.weight("key")) @ q.T                               # (a, b)
    s1 = ((v1 @ g.weight("key")) * q).sum(axis=1)                   # (b,)
    diff = s0 - s1[None, :]
    w0 = lang.softmax(lang.stack([diff, diff * 0.0], axis=2), axis=2)[:, :, 0]   # the first slot's share of the attention
    # v1[b] unembed + w0 t0[a] + (1 - w0) t1[b]
    logits = (v1 @ unembed + t1)[None, :, :] + w0[:, :, None] * (t0[:, None, :] - t1[None, :, :])
    two = (logits.argmax(axis=2) == lang.maximum(tokens[:, None], tokens[None, :])).mean()
    return (one + two) / 2.0


ESTIMATORS = {
    "tables over single tokens, every input": every_input,
}
