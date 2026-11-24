
The verifier scores an estimator on copies of the network (see "How an estimator is scored"), and
your sandbox makes and measures copies the same way:

```python
w = copy(seed=3)                   # the weights of a copy, made as the verifier makes its own; a dict like `weights`
a = accuracy(w)                    # the copy's accuracy, measured as the verifier measures it; accuracy() is your network's
g = graph(weights=w)               # a graph whose example run is on the copy
r = run(e, runs=50, weights=w)     # the estimator on the copy: an estimator names its weights, so it runs on any copy
out, ans = outputs(x, weights=w)   # the copy's outputs and the task's answers on inputs x
s = score_on_copies(e, copies=16)  # a whole scoring on 16 copies drawn from your seed: s["reward"],
                                   # s["mean_squared_error"], s["cost"], and per copy s["accuracy"], s["estimate_mean"]
```

`info["copies"]` holds the numbers: how many copies a scoring uses, the runs on each, and how a copy
is made.
