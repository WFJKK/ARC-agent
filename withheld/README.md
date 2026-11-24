# What is withheld from the agent

No tool of the agent reaches this folder, and nothing the agent can reach may say what is written here
(`arcagent/hygiene.py` checks it before every run and every prompt). Unlike in Mech-Interp-Agent-level1, the
agent is told the task: an accuracy means nothing without it, so `models/<name>/task.py` states it in the
estimator language and `config.yaml` in words. What is withheld is the answer and the way to it:

- how often each network gives the task's answer (the verifier measures it itself, on 2,097,152 inputs of its
  own, and the agent never sees that number);
- which network is which: the run's network is one of several, and the others are held out for the report;
- the names that lead to published figures (the source of the networks and its write-up).

| Model | Networks | Accuracy (verifier, 2^21 inputs) |
| --- | --- | --- |
| rnn-g | s0: ARC's AlgZoo challenge network, 432 weights | 0.953078 |
| rnn-g | z0 to z4: AlgZoo's own networks of the same size, five seeds | 0.910, 0.880, 0.949, 0.932, 0.890 |
| rnn-g | n1 to n3: s0 with noise of 0.2%, 0.5% and 1% of each matrix's RMS | 0.945, 0.846, 0.694 |
| rnn-b | s0 to s4: AlgZoo's networks of hidden size 4 on three numbers | 0.985, 0.893, 0.759, 0.904, 0.945 |

- `rnn/`: AlgZoo's original files and the converter that reads them without running them.
- `<model>/task.yaml`: the full record, with the strings the hygiene check looks for.
- `<model>/references.py`: reference estimators for the report.
- `rnn-g/noise.py`: how the noised networks were made.
