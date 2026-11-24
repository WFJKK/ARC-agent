# Design

What the pipeline does and why it is built this way. `docs/REWARD.md` covers the reward and how it could be gamed;
`docs/SANDBOX.md` covers where the agent's code runs. Most of the machinery is Mech-Interp-Agent-level1's; this
file says what is the same and what changed.

## 1. In a few lines

ARC asks for estimates of a network's behaviour that reason from the weights and compete with sampling at the same
compute. The pipeline keeps the competition and drops the reasoning requirement: an estimator is measured, not
judged, and anything that estimates well and cheaply scores well, sampling included.

- **Network** M with F operations per forward pass, **data set** D, **task** y.
- **Accuracy** A: the share of D on which M(x) = y(x).
- **Estimator** G (the agent's work): a program that computes one number from the weights, with random draws if
  it wants them, in a language of priced operations.
- **Verifier**: measures A itself, runs G many times with fresh seeds, and measures each run's error and cost.

## 2. The estimator

An estimator is a graph of operations on tensors stored as JSON data: `{"format": "arc-estimator-v1", "nodes":
[...], "estimate": n}`. The sources are the weights by name, constants, ranges and random draws; every other node
applies one operation to earlier nodes. The agent writes estimators in Python with a builder (`lang.Graph`), and
the verifier interprets the JSON. Nothing of an estimator is executed as code, so it can do no work that is not
counted.

**Prices** (`arc-prices-v1`, the table at the top of `arcagent/lang.py`, repeated for the agent in
`agent_docs/ESTIMATOR.md`): level 1's `prices-v1` with three additions. Random draws cost 1 per number drawn; `erf`
(for Gaussian probabilities, `lang.normal_cdf`) and `power` cost 1 per entry like the other elementwise
operations. The network is written in the same language (`models/<name>/model.py`, on a batch of inputs), so a
forward pass and an estimator are priced by one list. Every network's forward pass costs what it costs in level 1
(2,937 operations for the 432-weight RNN). For that, `take_sum` (a matrix applied to a sum of one-hot vectors) takes
a batch of index vectors, one sum per vector at the price of one.

**Random draws.** `g.normal(shape)`, `g.uniform(shape)`, `g.integers(high, shape)` and `g.inputs(n)` each make a
random node with a stream number of its own (an input with a varying number of tokens makes two: the tokens, then
their number, and the slots past the last token are emptied by a comparison and a where per slot). In the run whose seed is the list S, the node with stream s draws from
`numpy.random.default_rng(S + [s])`. Within one run a node is computed once, so two uses of it see the same draws;
two nodes are independent.

**Where the cost is charged.** A node that depends on no draw (a fixed node) has the same value in every run. The
interpreter computes it the first time it is needed and keeps it, but every run is charged for all the fixed work
it reads: each fixed node has a closure (the fixed nodes it took, with the chosen side of a `cond`), and a run pays
once for the union of the closures it reads. So a run costs exactly what it would cost alone, and precomputing from
the weights is paid in every run. `cond(c, x, y)` computes only the chosen side, so an estimator can spend on a
few runs and pay only there.

**Limits.** A run limits an estimator to 400 nodes and 256 numbers in constants (config: `entry`), as in level 1.
No value may hold more than 4,000,000 numbers and no product do more than 2^28 multiply-adds; a run may hold
64,000,000 numbers at once.

## 3. The model folder and the verifier

A model (`models/<name>/`) is `config.yaml` (sizes, the data set and the task, each with the sentence the agent is
told, the weights' names and shapes, the networks with their sha256), `model.py` (`logits` and `forward` on a batch,
in the language), `task.py` (`answer` on a batch, in the language) and `zoo/*.npz`. What the agent must not see is
in `withheld/<name>/`: how often each network gives the task's answer, which network is which, where they come
from, the strings the hygiene check looks for, and reference estimators.

**The task is told.** Level 1 never told the agent what the network was trained to do. Here an accuracy means
nothing without the task, so `task.py` and `task.says` state it, as a task: an answer for every input, with no word
on whether the network was trained for it, or on which inputs. What is withheld is the answer (A), where the
networks come from (how they were made, what they were trained on, if anything) and the route to published
figures.

**Data sets.** Three kinds (`lang.KINDS`): `all`, every combination of one token per position; `lengths`, a
number of tokens drawn first from a list, each as likely as the others, then the tokens, filling the first slots,
with the rest empty; `normal`, real numbers drawn from the standard normal distribution. A model may say
`max_batch`, the most inputs it is run on at once (the attention networks: 16,384, so that a batch's vectors stay
within the limit per value).

`arcagent/verifier.py` is the only judge. Given an estimator, it

1. checks that it is an estimator (shape, operations, limits) and refuses it with the reason if not;
2. measures A once: the network on `verifier.reference_inputs` (2,097,152) inputs drawn with
   `verifier.reference_seed`, through the same interpreter (`networks.Model.accuracy`), kept in the run folder; a
   data set with at most that many inputs is measured exactly instead, on every input, each weighted by how often
   it is drawn (method `exact`);
3. runs the estimator `verifier.runs` (4,096) times, run k with the seed `[seed, k]` for a seed drawn afresh from the
   operating system for each scoring; an estimator that draws nothing is run once and repeated;
4. returns a receipt: the parts (mean squared error, bias, spread, mean and range of the cost, draws), the reward
   under the run's reward function, and the benchmark (plain sampling at the same cost; the best plain sampling
   under the reward).

One more sampled input costs plain sampling S operations: the draws of one input, a forward pass, the task's answer
and a comparison (`Model.sample_ops`; 2,989 for the 432-weight RNN). The benchmark uses that price.

## 4. The agent's session

A round is one session with two tools (`arcagent/session.py`):

- `run_python(code)`: runs the agent's code in a sandbox. In scope: the weights, `lang`, `model`, `task`,
  `info` (sizes, the data set, F, the verifier's runs and limits), `graph()`, `run(estimator, runs, seed)` (the
  verifier's own code), `score_against(estimator, accuracy)` (the reward for an accuracy the agent supplies),
  `sample(n)` and `outputs(x)` (inputs and the network's outputs and the task's answers on them), the saved
  estimators and `save_estimator(name, estimator)`. Exploring costs nothing.
- `submit_estimator(name)`: hands a saved estimator to the verifier and returns `verifier.agent_view` of the
  receipt: the reward, the mean squared error, the cost, the mean and spread of the estimates, whether it draws.
  Not the accuracy, the bias or the benchmark.

Both have budgets per round (config: `session`). Every call is appended to the session's `tools.jsonl`, every
program is kept in full with what it printed, and every submission keeps its estimator, its full receipt and every
run's estimate and cost (`runs.npz`).

**What reaches the agent.** The prompt, the tools' descriptions and the files of its sandbox folder are searched
for the run config's and the model's forbidden strings (`arcagent/hygiene.py`) before the run starts and before
every prompt: the accuracies of the model's networks as numbers, the words that would say how the networks were
made, and the names that lead to the published figures. The tests also check that nothing the agent reaches
mentions training at all.
The sandbox keeps the agent's code from the repository, the withheld folder, the held-out networks' weights and the
run folder.

## 5. The rounds, the lock, the held-out check

`arcagent/loop.py` drives a run.

- **Before the first round**: A is measured, plain sampling written as an estimator must cost exactly what it
  draws, the hygiene check must pass, and the sandbox is probed with real escapes.
- **A round**: the prompt (`prompts/round_v1.md`) holds the network's and the task's definitions, the data set, the
  estimator guide, how estimators are scored, the budgets, and the earlier rounds: every submission's result as the
  agent saw it, and the notes the agent left.
- **The rounds end** at `rounds_max` (5), after `patience` (2) rounds in a row that did not raise the best reward,
  or after two refusals in a row.
- **The lock**: the scored submission with the highest reward, the earliest on a tie, byte for byte as submitted (its
  hash is checked). Its recorded reward is the best of several noisy scorings, so it is scored once more with a fresh
  seed, and the report gives both.
- **The held-out check**: the locked estimator, unchanged, on every network of `heldout.networks`, with the full
  number of runs. For the record, every scored submission is also scored there after each round
  (`heldout_scores.jsonl`, fewer runs, one fixed seed per network). The agent never sees these scores, and they
  choose nothing.
- **The report**: the accuracy, every submission with its parts and its mean held-out reward, the lock, the held-out
  check next to the constant of the run's own accuracy and to plain sampling at the same cost, the reference
  estimators, the frontier of mean squared error and cost, and the agent's closing notes.

## 6. Who answers a round

As in level 1: `backend: claude_code` runs the Claude Code CLI headless, once per round, with no built-in tools,
no files, no project settings and one tool server with the session's two tools. Model fallback is off; usage limits
are waited out; a login problem stops the run cleanly. `auth: login` uses your own login; `auth: api_key` bills
the key in `api_key_env` within a budget. `backend: fake` replays scripted tool calls (tests and dry runs).

## 7. Stopping and resuming

Everything a run knows is in its folder, written as it happens (append-only logs). A round is done when its
`round.json` is written; `resume` starts at the first round that is not, and a session that was cut off keeps its
budgets, its saved estimators and its submissions. If the code, the prompt or the model folder changed since the
run began, `resume` refuses unless told `--accept-changes`. `finish` redoes the lock, the held-out check and the
report from the records alone, and never measures again what it measured before.

## 8. Adding a model

1. `models/<name>/config.yaml`: sizes, the input space, the data set (one of the three kinds) and the task with
   their sentences, the weights' names and shapes, the networks (id, file, sha256), and `max_batch` if a batch of
   32,768 inputs would be too large. Its comments say nothing of where the networks come from.
2. `models/<name>/model.py`: `logits(g, lang, x)` and `forward(g, lang, x)` on a batch. Every input must cost the same
   (checked on batches of 1, 2 and 5).
3. `models/<name>/task.py`: `answer(g, lang, x)` on a batch, with its price per input in its docstring (the guide
   points the agent there).
4. `models/<name>/zoo/*.npz`: the weights, saved without pickle.
5. `withheld/<name>/task.yaml`: provenance, the verifier's accuracy of each network, and the hygiene strings: each
   accuracy to four places and as a percent to two, and where they do not round to 1, to three places and to one;
   `tests/test_hygiene.py` checks the check would catch every form. Optionally `references.py`.
6. A pilot in `config/pilot/<name>.yaml` with its held-out networks, and a `-smoke` version to try first.
7. The model's prices and task in `tests/test_models.py` (`PRICES`, `TASKS`), with a numpy version of a new
   architecture.

## 9. Defaults, and where to change them

| What | Default | Where |
| --- | --- | --- |
| Reward | `r_lambda`, lambda = 1e-6 | `config/defaults.yaml`, `reward` |
| Runs per scoring | 4,096 | `verifier.runs` |
| A | 2,097,152 inputs, seed 20261124; every input where the data set has no more | `verifier.reference_inputs`, `reference_seed` |
| Time per scoring | 600 s for all the runs | `verifier.time_limit_s` |
| Estimator limits | 400 nodes, 256 numbers in constants | `entry` |
| Rounds | at most 5, patience 2 | `rounds_max`, `patience` |
| Session budgets | 40 `run_python` calls, 4 submissions, 150 turns, 90 minutes | `session` |
| One `run_python` call | 300 s, 4 GB | `python` |
| Held out | every submission with 256 runs; the lock with 4,096 | `heldout` |
| Agent | `claude-fable-5-1`, effort high (level 1's pilots used this model) | `agent_model`, `effort` |
| Sandbox | `auto`: the strongest mode that works | `sandbox` |

Numbers live in the config files and never in code. Nothing specific to one model is in `arcagent/`.

## 10. What is not here

- A check that an estimator is mechanistic. ARC's estimators reason instead of sampling; here sampling is allowed,
  and only its price in the reward and the benchmark in the report tell the two apart.
- Surprise accounting, ARC's stricter measure.
- A baseline agent asked for the cheapest sampler, which would price sampling at what an agent can make it cost
  rather than at the plain sampler's price.
- Per-context accuracies (the accuracy given part of the input), which make a constant much harder to guess.
- A count of the forward passes the agent's own code makes while exploring. Its programs are kept, so the count can
  be made afterwards.
