"""Reference estimators for lin-a, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g, the estimator language lang, and the model's and the task's modules, and returns the
estimate as an expression.

The network adds one embedding column per token and applies two matrices, so the logits of an input are the sum
of one column per token of M = unembed hidden embed: logit j of the input (a, b) is M[j, a] + M[j, b]. The
estimators below work from that table instead of running the network. They are in the order they were found, each
tried because of what the one before cost or missed.
"""


def _token_logits(g):
    """M (outputs x tokens): what a token adds to each logit. d_model * d_model * tokens + outputs * d_model * tokens
    multiply-adds."""
    return g.weight("unembed") @ (g.weight("hidden") @ g.weight("embed"))


def sampled_from_the_table(n):
    """Plain sampling with the network replaced by its table: per input two columns of M added up, the highest logit,
    and the task's answer, about 0.3 forward passes in place of one. The table is paid in every run."""
    def build(g, lang, model, task):
        x = g.inputs(n)
        out = lang.take_sum(_token_logits(g), x, axis=1).argmax(axis=0)
        return (out == task.answer(g, lang, x)).mean()
    return build


def every_input(g, lang, model, task, block=4):
    """The accuracy itself, from the table: every pair of tokens, once. The two orders of a pair give the same
    logits and the same answer, so a pair of two different tokens is looked at once and counted twice. The tokens are
    gone through in blocks: the block's tokens b against every token a below the end of the block."""
    M = _token_logits(g).T  # (tokens, outputs)
    n = M.shape[0]
    tokens = g.arange(n)
    right = 0.0
    for lo in range(0, n, block):
        hi = lo + block
        logits = M[:hi][:, None, :] + M[lo:hi][None, :, :]   # (hi, block, outputs): tokens a < hi with the block's b
        out = logits.argmax(axis=2)
        if lo:
            # a below the block: the larger token is b, and the pair in its other order is not met again
            right = right + 2.0 * (out[:lo] == tokens[lo:hi][None, :]).sum()
        both = lang.maximum(tokens[lo:hi][:, None], tokens[lo:hi][None, :])  # a and b both in the block: every order
        right = right + (out[lo:] == both).sum()
    return right / float(n * n)


def sampled_in_permutations(k):
    """Sampling from the table, k * tokens inputs, drawn as k permutations: in each, every token is the first token
    once and the second token once. An input (a, p(a)) of a random permutation p is as likely as any other, so the
    estimate is unbiased, and what a single token does to the accuracy (a changed column of the table, a changed
    row) is in every permutation alike instead of being drawn by chance: about a quarter less variance than the
    same number of inputs drawn one by one."""
    def build(g, lang, model, task):
        M = _token_logits(g)
        n = M.shape[1]
        first = g.arange(n)
        right = 0.0
        for _ in range(k):
            second = lang.argsort(g.uniform([n]), axis=0)        # a random permutation of the tokens
            x = lang.stack([first, second], axis=1)
            out = lang.take_sum(M, x, axis=1).argmax(axis=0)
            right = right + (out == lang.maximum(first, second)).sum()
        return right / float(k * n)
    return build


def top_rows(K, factor=1.0, block=8):
    """Every input, but only the rows of the table that can win: for the input (a, b) the K highest rows of column a
    and the K highest of column b. The network's output maximizes M[j, a] + M[j, b] over the rows j, and a row that
    is high in neither column seldom does; so the estimator counts the inputs on which the larger token's logit is
    at least the best of those 2 K rows' logits. It is never below the accuracy (an input counted wrong is wrong)
    and misses the wrong inputs whose output is outside the 2 K rows: 1.4 points of accuracy on a copy for K = 2,
    0.5 for K = 4, 0.1 for K = 8. `factor` scales the error rate it finds to make up for those (1: no scaling).

    Per column a sort, per K one row of the table read for every pair, an addition and a maximum; no forward pass
    and no draw. The two orders of a pair are the same, so the count goes over a <= b, in blocks."""
    def build(g, lang, model, task):
        M = _token_logits(g)                       # (rows, tokens), square: a row per token value
        n = M.shape[1]
        order = lang.argsort(M, axis=0)            # per column, the rows from lowest to highest
        best = None
        for k in range(K):
            rows = lang.take(M, order[n - 1 - k], axis=0)      # [t, u] = M[the k-th highest row of column t, u]
            here = rows.reshape(n * n)[::n + 1]                # [t] = that row's entry in column t itself
            logit = rows + here[:, None]                       # that row's logit on the input (t, u)
            best = logit if best is None else lang.maximum(best, logit)
        best = lang.maximum(best, best.T)                      # the top rows of either token's column
        own = M.T + M.reshape(n * n)[::n + 1][None, :]         # [a, b] = M[b, a] + M[b, b]: the logit of row b
        twice = g.const([[2.0 if i < j else 1.0 if i == j else 0.0 for j in range(block)] for i in range(block)])
        right = 0.0
        for lo in range(0, n, block):
            hi = lo + block
            # a and b both in the block: the pairs with a < b count twice, a == b once
            right = right + ((own[lo:hi, lo:hi] >= best[lo:hi, lo:hi]) * twice).sum()
            if hi < n:   # a in the block, b above it: b is the larger token
                right = right + 2.0 * (own[lo:hi, hi:] >= best[lo:hi, hi:]).sum()
        share = right / float(n * n)
        return share if factor == 1.0 else 1.0 - factor * (1.0 - share)
    return build


ESTIMATORS = {
    "the table of token logits, 448 sampled inputs": sampled_from_the_table(448),
    "the table of token logits, every input": every_input,
    "the table of token logits, 6 permutations": sampled_in_permutations(6),
    "the top 4 rows of each token's column, every input": top_rows(4),
    # 1.098: the factor that fits the error rates of 150 copies made in a sandbox (seeds 1000 to 1149), measured by
    # going through every input. A number from brute force, on top of a count that follows the weights.
    "the top 3 rows of each token's column, every input, error rate times 1.098": top_rows(3, 1.098),
}
