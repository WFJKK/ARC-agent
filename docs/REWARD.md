# The reward

The verifier measures an estimator; a reward function turns what it measured into one number. The measuring is
fixed. The reward is meant to be changed: it is a function in `arcagent/rewards.py`, picked by the run config, and
every receipt keeps the parts, so a finished run can be scored again.

A scoring is made on copies of the run's network, not on the network itself (section 5 says why and how). Sections
1 to 3 are written for one network; on copies, A is the accuracy of the copy a run was on, and every mean is over
all runs on all copies.

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
(inputs, seed, standard error) and the benchmark (section 3).

On copies there are more parts: `copies` and `runs_per_copy`; `accuracy` and `accuracy_std`, the mean and the
standard deviation of the copies' accuracies; `estimate_std`, the spread of the estimates within a copy (the mean
of the copies' own standard deviations), and `estimate_between`, the standard deviation of the copies' mean
estimates; and `tracking`, the correlation between a copy's mean estimate and its accuracy (none for an estimator
that gives every copy the same number). The receipt keeps every copy's accuracy, mean estimate and seed.

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

On copies, A (1 - A) is its mean over the copies, and the benchmark has two more numbers: `constant_best`, what
the best number written into an estimator could score on these copies (minus the variance of their accuracies),
and `constant_unchanged`, what the network's own accuracy scores there.

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
| Send one estimator again and again and keep the luckiest scoring (the first attention pilot did, three times in its last round) | every submission of a run is scored with the run's one seed, on the same copies with the same draws, and an estimator the run has scored is refused without spending a submission; the lock is scored once more with a fresh seed, and the report gives that number |
| Read the accuracy from the receipts | the agent sees no accuracy, bias or benchmark. On one network the reward and the mse still say how far off a number is; on copies they say it of copies that are never used again |
| Measure the accuracy beforehand and write it in | scoring on copies: the number is 8 to 23 times worse than the best plain sampling there (section 5) |
| Learn the run's copies from what its scorings report, and answer for them | a scoring reports a handful of numbers over all the copies and no accuracy, a run has at most rounds times submissions of them, the seed stays in the run folder, and the lock's second scoring is on copies drawn after the choice |
| Carry the network's weights in as constants and look up, in a table made beforehand, what each changed weight does | nothing forbids it, and it loses to plain sampling: the changes of several weights do not add up (97 numbers for rnn-b's 32 weights; 1.1 to 3.5 times sampling's loss over five seeds) |

## 5. Scoring on copies: what the cost term cannot see, and what stops it

**The number.** The cost term makes brute force inside an estimator expensive. It cannot see brute force done
beforehand: the agent's code may run the network as often as it likes while exploring, and the accuracy it
measures there fits into an estimator as one constant, at no operations. Scored on the network itself that
constant is the best estimator there is (on the challenge network even 0.95, read off nothing, scores -9.5e-6, far
above any sampling), and no estimate can reach it.
`tests/test_gaming.py::test_the_known_limit_a_constant_read_off_the_network` records it.

**Copies.** So a scoring is made on copies of the network (`arcagent/measure.py`): each weight changed with
probability `share`, by a normal draw of `size` times the root mean square of its matrix; zeros stay zero. Every
scoring makes `count` copies of its own from its seed, measures each copy's accuracy, runs the estimator `runs`
times on each, and takes a run's error against the copy it ran on. What is computed from the weights follows the
copy; a number does not, and its mean squared error is at least the variance of the copies' accuracies.

**How much the copies differ** is therefore the one thing to choose, per network (`python -m arcagent.calibrate`):

- share: 1% of the weights, and at least 4 of them on average;
- size: the smallest at which the variance of the copies' accuracies is 4 times what the best plain sampling loses,
  2 sqrt(lambda s v), v the copies' mean A (1 - A). It must hold at the next size of the grid too.

A few weights changed by much, not all of them by little: with noise on every weight a wide network averages it
out, a copy's accuracy is nearly a function of the noise level alone, and a number (or a table of numbers by
noise level) would do again. The record of each calibration is in `withheld/<model>/copies.yaml`.

**What it costs.** The copies are not the network: their accuracies are lower and differ, and an estimator that
leans on one detail of the one network may fail where that detail was changed. The report gives the locked
estimator on the network itself as well, and `verifier.copies.count: 0` scores a whole run there.

**Sampling.** An estimator that samples well scores well. Variance reduction (stratifying, control variates,
importance sampling) is rewarded like understanding is. Telling the two apart is left to reading the estimators.
What copies add is that there is something above sampling to reach for: the reference estimators of lin-a, att-a
and rnn-a compute the accuracy from the weights and lose a little over a third, under a quarter, and a fourteenth
of what the best plain sampling loses (README, "Scored on copies of the network").

**Held-out networks** are other networks of the same design, scored as they are. They stay in the report, where
they say whether the locked estimator works beyond the run's network at all.

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
- **How the copies are made** is not the reward but the measuring: `verifier.copies` in the run config. Runs made
  with different shares or sizes cannot be compared with each other.
