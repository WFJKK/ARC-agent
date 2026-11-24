# The reward

The verifier measures an estimator; a reward function turns what it measured into one number. The measuring is
fixed. The reward is meant to be changed: it is a function in `arcagent/rewards.py`, picked by the run config, and
every receipt keeps the parts, so a finished run can be scored again.

## 1. The parts

With A the network's accuracy as the verifier measures it, K runs of the estimator, run k giving the estimate
A_k and spending O_k operations, and F the operations of one forward pass:

| Part | Computed as |
| --- | --- |
| mse | mean over k of (A_k - A)^2 |
| rms_error | sqrt(mse) |
| bias | mean over k of A_k, less A |
| estimate_std | the standard deviation of the A_k |
| cost | mean over k of O_k / F, in forward passes |
| cost_min, cost_max | the least and most O_k / F |
| draws_mean | the random numbers a run draws, on average |
| random | whether the estimate depends on a draw at all |

mse is bias^2 + estimate_std^2. A receipt also keeps the seed, the number of runs, the reference measurement
(inputs, seed, standard error) and the benchmark (section 4).

**A is itself measured.** On 2,097,152 inputs its standard error is about 0.015 points for a network near 95%, so an
estimator's squared error has a floor of about 2e-8 from the reference alone. That is 0.02 forward passes at lambda =
1e-6: nothing for an estimator that runs the network, but it is the whole reward of a constant equal to the
measured A.

## 2. `r_lambda`, the default

    reward = mean over runs of [ -(A_k - A)^2 - lambda * C_k ] = -mse - lambda * cost,   lambda = 1e-6

The squared error and the compute are traded at the rate lambda. At 1e-6 one point of error (0.01^2 = 1e-4) weighs
as much as 100 forward passes, so 100 more forward passes that remove one point of error break even. 0 is the most
there is: the exact accuracy at no cost.

**Lambda is a choice of sampling budget.** Plain sampling with n inputs has mse A(1 - A)/n and costs n s forward
passes, where s is what one more sampled input costs (1.0177 for the RNN: ten draws, a forward pass, the task's
answer and a comparison). Its reward is best at

    n* = sqrt(A (1 - A) / (lambda s)),   reward* = -2 sqrt(lambda s A (1 - A))

For the challenge network at lambda = 1e-6 that is 210 inputs and -4.27e-4. Choosing lambda is choosing the budget
at which an estimator has to compete: a smaller lambda asks for more precision and allows more compute.

**The worked example** (README): an estimator with rms error 1.286 points at 109.86 forward passes scores
-(0.0128565)^2 - 1e-6 * 109.86 = -2.7515e-4, above the best plain sampling's -4.27e-4. `tests/test_rewards.py` checks
the arithmetic.

## 3. The benchmark

For every receipt, never shown to the agent: plain sampling at the estimator's mean cost C affords n = C / s inputs,
and its mse is A (1 - A) / n. The receipt gives that mse and rms error, the share by which the estimator's mse is
lower (`mse_reduction`: 59.9% in the worked example), and the best plain sampling under the reward (n* and
reward*). Below one sample (C < s) sampling has nothing to say and these are empty.

## 4. Ways to game the reward, and what stops each

Each has a test in `tests/test_gaming.py`, unless another test is named or it holds by construction. A hole found
later gets its test there first.

| Route | What stops it |
| --- | --- |
| Do work outside the count: code, files, other weights | an estimator is data; only the listed operations and the run network's weights exist |
| Precompute from the weights once and reuse it in every run for free | every run is charged for all the fixed work it reads (`tests/test_lang.py` checks a run costs what it costs alone) |
| Charge an expensive branch as a cheap one | a `cond` branch is charged in every run that takes it |
| Fit to the verifier's draws | a fresh seed per scoring, drawn by the verifier; the reference inputs come from a seed no scoring uses |
| Draw once and reuse the draw as many samples | it is one draw: the run is charged for it, and the estimate is as good as one sample |
| Paste a table of answers | the limit of 256 numbers in constants |
| An estimate that is not a number on rare runs | refused, with the run and its seed |
| Send one estimator again and again and keep the luckiest scoring | the lock is scored once more with a fresh seed, and the report gives that number |
| Read the accuracy from the receipts | the agent sees no accuracy, bias or benchmark; the reward and the mse still say how far off it is, which a reward on one network cannot avoid |

## 5. What the reward does not see

**The constant.** The reward scores only the estimate and its cost on the one network of the run. The agent's code
may run the network as often as it likes while exploring, so the accuracy measured in the sandbox and written in as
a constant scores near the top. On the challenge network the constant 0.95, read off nothing, scores -9.5e-6, far
above any sampling. This is not a hole in the measuring but the reward's definition, and the pipeline keeps it as
the default because it is the quantity the write-up asks about. What the pipeline adds instead:

- the agent is told that the locked estimator is run on other networks, where a constant stays what it was;
- the held-out check and the per-submission held-out scores (`heldout_scores.jsonl`) measure it on networks of the
  same design: five trained from other starts and three noised copies of the challenge network, with accuracies
  from 69% to 95%;
- the report sets every held-out result next to what the run's own accuracy, as a constant, would score there.

`tests/test_gaming.py::test_the_known_limit_a_constant_read_off_the_network` records it: the pasted accuracy beats
sampling on the run's network and loses to it on another.

**Sampling.** An estimator that samples well scores well. Variance reduction (stratifying, control variates,
importance sampling) is rewarded like understanding is. Telling the two apart is left to reading the estimators
and to the held-out check, where a mechanism that holds keeps working and a tuned sampler mostly does too.

## 6. Changing the reward

- **A parameter**: in the run config, `reward: {lam: 1.0e-7}` (write the number with a decimal point: YAML reads
  `1e-7` as text).
- **A new reward**: a function in `arcagent/rewards.py` that takes the parts and its own parameters by name and
  returns at least `{"reward": ...}`; add it to `REWARDS`, and its formula and explanation for the agent's prompt to
  `FORMULAS` and `EXPLANATIONS`. The file is placed in the agent's sandbox, so its texts pass the hygiene check. A
  parameter the function does not take is refused.
- **See what a reward does before running anything**: `python -m arcagent.report --model models/rnn-g --lam 1e-7`
  scores the references.
- **Score a finished run again**: `python -m arcagent.loop rescore --run <run folder> --lam 1e-7`. The records are not
  changed; the agent was steering by the reward the run was made with.
- **A reward on the held-out networks**: the held-out scores of every submission are on record, so a reward that
  averages over networks can be tried on a finished run first (`heldout_scores.jsonl`). To steer by it, the
  verifier would score each submission on those networks and the agent would be scored on networks it cannot see.
