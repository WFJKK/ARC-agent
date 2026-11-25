The verifier does not score your estimator on the network as you hold it. It scores it on
{{COPIES}} copies of the network, each with a few of its weights changed at random. On a copy, `g.weight(...)` gives the copy's weights, so what your estimator
computes from the weights follows the copy; a number you measured on your network and wrote into the
estimator stays what it was.

A copy is made like this. Every weight is changed with probability {{COPY_SHARE}}, each on its own:
a number is added to it, drawn from a normal distribution whose standard deviation is {{COPY_SIZE}}
times the root mean square of the entries of its weight matrix. The other weights stay exactly as
they are, and a weight that is exactly zero is never changed. `copy(seed)` in your sandbox makes a
copy this way. The verifier draws its copies once for this run, from a seed it keeps to itself, so
you never see them. Every submission of the run, in every round, is scored on those same copies with
the same random draws: two estimators' scores differ by what the estimators compute and by nothing
else, and an estimator that has been scored is not scored again (the call is refused and costs no
submission). When the rounds are over, the best estimator is scored once more on fresh copies, and
that score is the one that is reported.

For each copy the verifier measures the copy's accuracy A itself: it runs the copy on
{{COPY_REFERENCE}} and counts how often the output is the task's answer. You are not told any A. It
then runs your estimator {{COPY_RUNS}} times on the copy, each run with fresh random draws from a seed
it draws itself, and keeps each run's estimate and its cost in forward passes (its operations over
{{MODEL_OPS}}). That is {{RUNS}} runs in all. An estimator that draws nothing gives one number for each
copy, at one cost.
