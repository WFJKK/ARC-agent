"""Reference estimators for rnn-g, for the report. Withheld from the agent, like everything here.

Each is a function (g, lang, model, task) -> the estimate, built on the network's weights. The report scores
every one under the run's reward and conditions. Plain sampling at several budgets is added for every model by
the report itself (arcagent/report.py), so it is not here.
"""

ESTIMATORS = {
    # The constant guess of the reward's write-up: 95%, read off nothing, at no cost. On rnn-g-s0 it misses by
    # about 0.3 percentage points and scores above the sampling estimators; on the held-out networks it misses
    # by up to 26 points.
    "constant 0.95": lambda g, lang, model, task: g.const(0.95),
}
