An estimator is a program whose one result is a number: its estimate of the network's accuracy, the
share of the data set's inputs on which the network's output is the task's answer.

An estimator is a graph of operations on tensors, stored as JSON data. You write it in Python with the
estimator language `lang`, and the verifier interprets the JSON; nothing of an estimator is run as
code, so only the operations below exist.

### Writing one

```python
g = graph()                        # a new lang.Graph on this network's weights
R = g.weight("recurrent")          # a weight matrix, by name
c = g.const([1.0, 2.0])            # a constant; plain Python numbers and numpy arrays become constants too
x = g.inputs(100)                  # 100 inputs drawn from the data set, fresh in every run: (100, numbers per input)
z = g.normal([5, 3])               # standard normal numbers, fresh in every run; also g.uniform, g.integers
right = model.forward(g, lang, x) == task.answer(g, lang, x)    # the network and the task on those inputs
e = g.entry(right.mean())          # the estimator as JSON data: here plain sampling of 100 inputs
save_estimator("sampling_100", e)  # keep it; submit_estimator takes the name
r = run(e, runs=200)               # 200 runs as the verifier runs them: r["estimates"], r["cost"], r["ops"]
print(r["estimates"].mean(), r["estimates"].std(), r["cost"].mean())
```

For exploring in plain numpy, `weights` holds the weights as arrays, `sample(n, seed)` draws inputs
from the data set and `outputs(x)` gives the network's outputs and the task's answers on them.

Every expression is also computed in an example run while you build (seed 0, or `graph(seed=...)`),
so `x.shape` and `x.value` exist and a wrong shape fails at the line that makes it.
`lang.Graph.load(e, weights, info["data_set"], info["input_space"])` opens a saved estimator again and
returns `(graph, estimate)`. `lang.show(e)` prints an estimator one node per line.

Values are whole numbers, floats or booleans (64 bits). There are no loops and no Python `if` on
expressions: use `lang.where(c, x, y)` (both sides are computed) or `lang.cond(c, x, y)` (c is one
boolean, and only the chosen side is computed and charged). A Python loop that builds nodes is fine:
it runs while you write the estimator, and every node it makes is counted.

### Runs and random draws

The verifier runs an estimator many times. In each run, every random node draws afresh from the
run's own seed; within one run a node is drawn once, however often it is used, and two random nodes
are independent. `g.inputs`, `g.normal`, `g.uniform` and `g.integers` each make a new random node.
What does not depend on a random draw is the same in every run, and every run is charged for all of
it that the run uses: precomputing from the weights is paid in every run, not once.

`run(e, runs, seed)` runs the estimator exactly as the verifier does (run k with the seed `[seed, k]`)
and `score_against(e, accuracy)` gives the reward it would get if the network's accuracy were the
number you give.

### Operations and what they cost

Each operation on numbers costs 1. Per entry of the result unless said otherwise:

| Operation | Cost |
| --- | --- |
| `+ - * / // % **`, unary `-`, `abs(x)`, `lang.maximum`, `lang.minimum`, `lang.exp`, `lang.log`, `lang.sqrt`, `lang.erf` | 1 |
| `< <= > >= == !=`, `&`, `\|`, `~` (booleans) | 1 |
| `lang.where(c, x, y)` | 1 |
| `lang.cond(c, x, y)` | 1 in all, plus the chosen side |
| `x @ y`, `lang.matmul(x, y)` | 1 per multiply-add: result entries times the contracted length |
| `x.sum(axis)`, `.max`, `.min`, `.argmax`, `.argmin`, `.any`, `.all` over n entries | n - 1 |
| `x.mean(axis)` | a sum and a division |
| `lang.cumsum(x, axis)` | n - 1 per vector |
| `lang.sort(x, axis)`, `lang.argsort(x, axis)` of n entries | n * ceil(log2 n) per vector |
| `lang.softmax(x, axis)` | 3 |
| `lang.normal_cdf(x)`: P(Z <= x) for a standard normal Z | 4: a product, erf, a sum and a product |
| `x[i]`, `x[i, j]`, `x[:, i]` with expressions that are one whole number (`lang.gather`) | 1 per entry read |
| `x[v]` or `lang.take(x, v, axis)` with a vector of whole numbers | 1 per entry read |
| `lang.take_sum(x, v, axis)`: the sum of the slices of x at the positions v | slice entries times len(v) |
| `lang.svd(x)` of an m x n matrix: `(u, s, vt)` | 4 * m * n * min(m, n), once for the three |
| `lang.cast(x, "int")`, `"float"`, `"bool"` | 1 |
| `g.inputs(n)`, `g.normal(shape)`, `g.uniform(shape)`, `g.integers(high, shape)` | 1 per number drawn |
| `g.weight`, `g.const`, `g.arange` | 0 |
| `x.reshape(...)`, `x.T`, `lang.transpose`, `x[2]`, `x[1:5]`, `x[None]`, `lang.concat`, `lang.stack` | 0 |

Reading costs 1 per number read: `T[a, b]` reads one number, `T[:, a]` a whole column. The network
on n inputs, `model.forward(g, lang, x)`, costs n forward passes; `task.answer(g, lang, x)` costs what
its sort costs.

### What an estimator's cost is

A run's cost is all the operations it computes, divided by the operations of one forward pass of the
network. The verifier's cost is the mean over its runs. `run(e)["cost"]` gives it per run.

### Limits

The run limits an estimator's size and the time the verifier gives it: the numbers are under "Tools,
budgets and limits" in your message. A constant counts every number it holds; no single value may
hold more than 4,000,000 numbers, nor one matrix product do more than 2^28 multiply-adds. The
verifier refuses an estimator, and says why, when in any run an index is out of range, shapes do not
fit, the result is not one finite number, a float that is not finite is cast to a whole number, or the
runs take longer than the time it gives them. It tells you the run and its seed, so that `run` can
repeat it.
