# What is withheld from the agent

No tool of the agent reaches this folder, and nothing the agent can reach may say what is written here
(`arcagent/hygiene.py` checks it before every run and every prompt). Unlike in Mech-Interp-Agent-level1, the
agent is told the task: an accuracy means nothing without it, so `models/<name>/task.py` states it in the
estimator language and `config.yaml` in words. What is withheld is the answer and the way to it:

- how often each network gives the task's answer (the verifier measures it itself, on 2,097,152 inputs of its
  own or on every input of a data set that has no more, and the agent never sees that number);
- which network is which: the run's network is one of several, and the others are held out for the report;
- where the networks come from: how they were made, which inputs they were trained on, or that they were never
  trained, and the names that lead to published figures (the source of the networks and its write-up).

Every model has a `task.yaml` here with the verifier's accuracy of each of its networks. rnn-g and rnn-b:

| Model | Networks | Accuracy (verifier, 2^21 inputs) |
| --- | --- | --- |
| rnn-g | s0: ARC's AlgZoo challenge network, 432 weights | 0.953078 |
| rnn-g | z0 to z4: AlgZoo's own networks of the same size, five seeds | 0.910, 0.880, 0.949, 0.932, 0.890 |
| rnn-g | n1 to n3: s0 with noise of 0.2%, 0.5% and 1% of each matrix's RMS | 0.945, 0.846, 0.694 |
| rnn-b | s0 to s4: AlgZoo's networks of hidden size 4 on three numbers | 0.985, 0.893, 0.759, 0.904, 0.945 |

The other eighteen are Mech-Interp-Agent-level1's models (commit `bee70fb`), the same files: the README's table of
models has each one's accuracies and origin, and level 1's `withheld/<model>/task.yaml` the whole record (training
recipes and curves, where a network differs from the task, how often it gives each answer).

- `rnn/`: AlgZoo's original files and the converter that reads them without running them; rnn-a, rnn-d, rnn-e and
  rnn-f come from the same folder (the untrained ones are AlgZoo's saved starting states).
- `<model>/task.yaml`: the record, with the strings the hygiene check looks for.
- `<model>/copies.yaml`: what `python -m arcagent.calibrate` measured when it chose how the run network's copies
  are made (the share and the size themselves are in the pilot config, and the agent is told them): how often
  copies give the task's answer, size by size.
- `<model>/references.py`: reference estimators for the report. Those of lin-a, att-a and rnn-a compute the
  accuracy from the weights, each from how its network works; they are what an agent is hoped to find, which is
  why they are here.
- `rnn-g/noise.py`: how the noised networks were made.
