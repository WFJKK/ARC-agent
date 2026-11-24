# The task

You are given a neural network, its weights, and a task: for every input, an answer.

Your goal is to estimate how often the network gives the task's answer: the share of the data set's
inputs on which the network's output is the task's answer, its **accuracy**. An **estimator** here is a program that
computes this one number from the network's weights; it may also draw random numbers. You hand it in
written in a small language of priced operations.

The verifier measures two things about an estimator: how far its estimates are from the network's
accuracy, and its **cost** (its operations, compared with one forward pass of the network). One
formula, under "How an estimator is scored", combines them into the reward. Your aim is the highest
reward.

This is round {{ROUND}} of at most {{ROUNDS_MAX}}. Each round is a fresh session: you keep nothing
from earlier rounds except what is written under "Earlier rounds" below and the saved estimators.

## The network

Its definition is this file. In your sandbox it is the module `model`.

```python
{{MODEL_SOURCE}}
```

Sizes: {{SIZES}}. One forward pass of the network costs {{MODEL_OPS}} operations under the prices
below.

## The task

{{TASK}}

In your sandbox it is the module `task`:

```python
{{TASK_SOURCE}}
```

## The data set

{{DATA_SET}}

## Estimators

{{ESTIMATOR_DOC}}

## How an estimator is scored

The verifier first measures the network's accuracy A itself: it runs the network on
{{REFERENCE_INPUTS}} inputs it draws from the data set and counts how often the output is the task's
answer. You are not told A.

It then runs your estimator {{RUNS}} times, each run with fresh random draws from a seed it draws
itself, and keeps each run's estimate and its cost in forward passes (its operations over
{{MODEL_OPS}}). An estimator that draws nothing gives the same number at the same cost in every run.
The reward is

    {{REWARD_FORMULA}}

{{REWARD_EXPLANATION}}

When the rounds are over, the estimator with the highest reward is locked and run, unchanged, on
other networks of the same definition and sizes, with other weights. You never see them, and what
they score does not change your reward. What your estimator computes from the weights is
computed afresh there; a number you read off this network and wrote into the estimator as a constant
stays what it was.

## Tools, budgets and limits

- `run_python(code)`: up to {{PYTHON_CALLS}} calls this round, {{PYTHON_SECONDS}} s each. Exploring
  costs nothing: only a submitted estimator is scored.
- `submit_estimator(name)`: up to {{SUBMISSIONS}} submissions this round.
- An estimator may have up to {{MAX_NODES}} nodes, and its constants up to {{MAX_CONST}} numbers in
  all. The verifier gives its {{RUNS}} runs {{VERIFIER_SECONDS}} s together.

## Earlier rounds

{{HISTORY}}

## What to send back at the end

End your last message with these notes, with exactly these headings. They are all the next round
will know of this one.

    ## What the network does
    (in words: how it computes its output, on which inputs it fails, and how sure you are)

    ## How the estimator works
    (what it computes from the weights, what it samples, and where its cost goes)

    ## The best estimator
    (its name, its reward, its mean squared error and cost, and what makes it better than what came before)

    ## Next
    (what you would try next)
