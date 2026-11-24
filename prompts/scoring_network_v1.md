The verifier first measures the network's accuracy A itself: it runs the network on
{{REFERENCE_INPUTS}} and counts how often the output is the task's answer. You are not told A.

It then runs your estimator {{RUNS}} times, each run with fresh random draws from a seed it draws
itself, and keeps each run's estimate and its cost in forward passes (its operations over
{{MODEL_OPS}}). An estimator that draws nothing gives the same number at the same cost in every run.
