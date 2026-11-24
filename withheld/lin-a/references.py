"""Reference estimators for lin-a, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g, the estimator language lang, and the model's and the task's modules, and returns the
estimate as an expression.

The network adds one embedding column per token and applies two matrices, so the logits of an input are the sum
of one column per token of M = unembed hidden embed: logit j of the input (a, b) is M[j, a] + M[j, b]. Both
estimators below work from that table instead of running the network.
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


ESTIMATORS = {
    "the table of token logits, 448 sampled inputs": sampled_from_the_table(448),
    "the table of token logits, every input": every_input,
}
