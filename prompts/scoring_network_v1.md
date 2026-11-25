The verifier first measures the network's accuracy A itself: it runs the network on
{{REFERENCE_INPUTS}} and counts how often the output is the task's answer. You are not told A.

It then runs your estimator {{RUNS}} times, each run with fresh random draws from a seed it keeps to
itself, and keeps each run's estimate and its cost in forward passes (its operations over
{{MODEL_OPS}}). The seed is the same for every submission of this run, so an estimator that has been
scored is not scored again (the call is refused and costs no submission); when the rounds are over,
the best estimator is scored once more with a fresh seed, and that score is the one that is reported. An estimator that draws nothing gives the same number at the same cost in every run.
