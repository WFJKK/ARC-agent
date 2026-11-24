"""Reference estimators for att-a, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g, the estimator language lang, and the model's and the task's modules, and returns the
estimate as an expression. They are in the order they were found.

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


def stratified(n):
    """Half the data set is the inputs of one token, and there are only as many of those as token values: the
    network on every one of them, and on n sampled inputs of two tokens. Sampling's variance is left on one half
    only, which counts a quarter."""
    def build(g, lang, model, task):
        tokens = g.arange(g.weight("embed").shape[0] - 1)
        one = lang.stack([tokens, tokens * 0 + g.data_set["empty"]], axis=1)
        two = g.integers(tokens.shape[0], [n, 2])
        right = lambda x: (model.forward(g, lang, x) == task.answer(g, lang, x)).mean()
        return (right(one) + right(two)) / 2.0
    return build


def _shared(g, lang):
    """The tables again, built once for both slots: a slot's vector is the token's embedding plus the slot's
    position, so a table over the tokens of a slot is one table over the embeddings plus one row for the position.

        logits of one token a:   EG[a] + c0,              EG = EU + ET, EU = embed unembed, ET = embed value out unembed
        of two tokens (a, b):    P[b] + w0 (t0[a] - t1[b]),   P = EU + t1 + pos[1] unembed, t_i = ET + pos[i] value out unembed
        attention:               s0 - s1 = S[a, b] - S[b, b] + k[b],   S = embed KQ v1', KQ = key query' / sqrt(d)

    Three products of the size (tokens, d, outputs) instead of five."""
    embed, pos, U = g.weight("embed"), g.weight("pos"), g.weight("unembed")
    n, d = embed.shape[0] - 1, pos.shape[1]
    E = embed[:n]
    VO = g.weight("value") @ g.weight("out")
    EU, ET = E @ U, (E @ VO) @ U
    pU, pT = pos @ U, (pos @ VO) @ U                              # (slots, outputs) each
    t0, t1 = ET + pT[0][None, :], ET + pT[1][None, :]
    one = EU + t0 + pU[0][None, :]                                # the logits of the one-token inputs
    P = EU + t1 + pU[1][None, :]                                  # of (a, b) if the second slot attends to itself only
    KQ = (g.weight("key") @ g.weight("query").T) * (1 / math.sqrt(d))
    v1 = E + pos[1]
    S = (E @ KQ) @ v1.T                                           # [a, b] = embed[a] KQ v1[b]
    diff = S - (S.reshape(n * n)[::n + 1] - ((pos[0] - pos[1]) @ KQ) @ v1.T)[None, :]      # s0 - s1 of (a, b)
    e = lang.exp(-abs(diff))                                      # the first slot's share of the attention, 1 / (1 + exp(-diff)),
    share = 1.0 / (1.0 + e)                                       # written so that nothing overflows
    w0 = lang.where(diff >= 0, share, 1.0 - share)
    return n, one, P, t0, t1, w0


def every_input_shared(g, lang, model, task):
    """The accuracy itself, like every_input, from the shared tables."""
    n, one, P, t0, t1, w0 = _shared(g, lang)
    tokens = g.arange(n)
    right_one = (one.argmax(axis=1) == tokens).mean()
    logits = P[None, :, :] + w0[:, :, None] * (t0[:, None, :] - t1[None, :, :])
    right_two = (logits.argmax(axis=2) == lang.maximum(tokens[:, None], tokens[None, :])).mean()
    return (right_one + right_two) / 2.0


def candidates(K):
    """Every input, but on two tokens only the outputs that can win. The logits of (a, b) lie between P[b] (all
    attention on b) and P[b] + t0[a] - t1[b] (all on a), so the output is nearly always one of the K highest
    outputs of P[b] or of t0[a]: the estimator counts the pairs on which the larger token's logit is at least the
    logits of those 2 K outputs, at the pair's own attention. Never below the accuracy; what it misses is the wrong
    pairs whose output is none of the 2 K (K = 1: 0.3 points of accuracy on a copy, K = 2: under 0.1). Per pair and
    candidate a read or two, a subtraction, a multiplication and an addition, instead of those for every output."""
    def build(g, lang, model, task):
        n, one, P, t0, t1, w0 = _shared(g, lang)
        tokens = g.arange(n)
        flat = g.arange(n * n).reshape(n, n)[:, 0]               # row t of a table starts at t * n in the flat table
        right_one = (one.argmax(axis=1) == tokens).mean()

        def top(X):          # per row of X, the positions of its K highest entries, highest first
            out, X = [], X
            for k in range(K):
                j = X.argmax(axis=1)
                out.append(j)
                if k + 1 < K:
                    X = lang.where(tokens[None, :] == j[:, None], -1e30, X)
            return out

        best = None
        for j in top(P):     # an output chosen by b: [a, b] = the logit of output j[b] on (a, b)
            Pj = lang.take(P.reshape(n * n), flat + j, axis=0)                 # P[b, j[b]]
            t1j = lang.take(t1.reshape(n * n), flat + j, axis=0)               # t1[b, j[b]]
            t0j = lang.take(t0.T, j, axis=0).T                                 # t0[a, j[b]]
            logit = Pj[None, :] + w0 * (t0j - t1j[None, :])
            best = logit if best is None else lang.maximum(best, logit)
        for j in top(t0):    # an output chosen by a
            t0j = lang.take(t0.reshape(n * n), flat + j, axis=0)               # t0[a, j[a]]
            Pj = lang.take(P.T, j, axis=0)                                     # P[b, j[a]]
            t1j = lang.take(t1.T, j, axis=0)                                   # t1[b, j[a]]
            best = lang.maximum(best, Pj + w0 * (t0j[:, None] - t1j))
        diag = lambda X: X.reshape(n * n)[::n + 1]
        first = P.T + w0 * (diag(t0)[:, None] - t1.T)                          # the logit of output a on (a, b)
        second = diag(P)[None, :] + w0 * (t0 - diag(t1)[None, :])              # the logit of output b
        own = lang.where(tokens[:, None] >= tokens[None, :], first, second)   # of the larger token
        right_two = (own >= best).mean()
        return (right_one + right_two) / 2.0
    return build


ESTIMATORS = {
    "every one-token input and 120 sampled inputs of two tokens": stratified(120),
    "tables over single tokens, every input": every_input,
    "tables shared between the slots, every input": every_input_shared,
    "tables shared between the slots, the 2 highest outputs of each token": candidates(2),
}
