"""The estimator language: programs in counted operations that compute one number.

An estimator is a small program, stored as JSON data, whose one result is a number: its estimate of how
often the network gives the task's answer. It is a graph of operations on tensors. The sources are the
network's weights (by name), constants, ranges and random draws; every other node applies one operation
to earlier nodes. Nothing in an estimator is code: the verifier reads the JSON and interprets it with the
table below, so an estimator can do no work that is not counted.

Prices (PRICES = "arc-prices-v1"): each operation on numbers counts 1. Per result entry unless said
otherwise:

    add sub mul div floordiv mod power neg abs maximum minimum exp log sqrt erf   1
    lt le gt ge eq ne and or not                                                1
    where(c, x, y)        both sides are computed                               1
    cond(c, x, y)         only the chosen side is computed                      1 in all
    matmul                one per multiply-add: (result entries) x (the contracted length)
    sum max min argmax argmin any all over n entries                            n - 1
    cumsum                                                                      n - 1 per vector
    sort argsort          of n entries                                          n * ceil(log2 n) per vector
    softmax                                                                     3
    take(x, idx, axis)    reading numbers                                       1 per entry read
    gather(x, i.., axes)  reading at scalar positions                           1 per entry read
    take_sum(x, idx, axis)  the sum of the slices at idx                        (slice entries) x len(idx)
    svd_u svd_s svd_vt    of an m x n matrix, charged once for the three        4 * m * n * min(m, n)
    cast                  to whole numbers, floats or booleans                  1
    normal uniform integers   a random draw                                     1 per number drawn
    weight const arange reshape transpose slice concat stack                    0

These are the prices of the language's first version (prices-v1) with three additions: random draws, erf
and power. The network itself is written in the same language (models/<name>/model.py), so one forward pass
and an estimator are priced by one list, and an estimator's cost is its operations over one forward pass.

**Runs.** The verifier runs an estimator many times, each run with fresh random draws, and charges every
run for everything it computes. A run's draws come from its own seed: a random node with stream s, in the
run whose seed is the list of whole numbers S, draws from numpy.random.default_rng(S + [s]). Within one
run a node is computed once, so two uses of one node see the same draws; two random nodes are
independent. What does not depend on a random draw is the same in every run, so the interpreter computes
it once and keeps it; a run is still charged for all of it that the run uses (Evaluator), so every run
costs exactly what it would cost on its own.

The same file gives the builder that writes estimators (Graph and its expressions), the interpreter
(Evaluator) and run_estimator(), which the verifier and the agent's sandbox both use. It needs numpy and
the standard library only.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import time

import numpy as np

FORMAT = "arc-estimator-v1"
PRICES = "arc-prices-v1"

MAX_NODES = 2000  # absolute ceilings; a run's own limits are lower (config: entry.max_nodes, entry.max_const_scalars)
MAX_CONST_SCALARS = 1_000_000
MAX_ELEMENTS = 4_000_000  # entries of any one value
MAX_WORK = 2**28  # multiply-adds of any one matrix product
MAX_HELD = 64_000_000  # entries of all the values held at once (what is kept between runs, and one run's own)
MAX_RANK = 6
MAX_STREAM = 2**62
SVD_FACTOR = 4

ELEMENTWISE_2 = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.true_divide, "floordiv": np.floor_divide,
    "mod": np.mod, "power": np.power, "maximum": np.maximum, "minimum": np.minimum, "lt": np.less,
    "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal,
    "and": np.logical_and, "or": np.logical_or,
}
_ERF = np.frompyfunc(math.erf, 1, 1)
ELEMENTWISE_1 = {"neg": np.negative, "abs": np.abs, "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                 "erf": lambda x: _ERF(x).astype(np.float64), "not": np.logical_not}
REDUCE = {"sum": np.sum, "max": np.max, "min": np.min, "argmax": np.argmax, "argmin": np.argmin, "any": np.any, "all": np.all}
ALONG = ("cumsum", "sort", "argsort", "softmax")
SVD = ("svd_u", "svd_s", "svd_vt")
SOURCES = ("weight", "const", "arange")
RANDOM = ("normal", "uniform", "integers")
DTYPES = {"int": np.int64, "float": np.float64, "bool": np.bool_}

ARITY = {**{k: 2 for k in ELEMENTWISE_2}, **{k: 1 for k in ELEMENTWISE_1}, **{k: 1 for k in REDUCE},
         **{k: 1 for k in ALONG}, **{k: 1 for k in SVD}, "where": 3, "cond": 3, "matmul": 2, "take": 2,
         "take_sum": 2, "gather": None, "reshape": 1, "transpose": 1, "slice": 1, "concat": None, "stack": None,
         "cast": 1, **{k: 0 for k in SOURCES}, **{k: 0 for k in RANDOM}}
ATTRS = {"weight": {"name"}, "const": {"value", "dtype"}, "arange": {"n"}, "take": {"axis"},
         "take_sum": {"axis"}, "gather": {"axes"}, "reshape": {"shape"}, "transpose": {"perm"}, "slice": {"index"},
         "concat": {"axis"}, "stack": {"axis"}, "cast": {"dtype"}, "normal": {"shape", "stream"},
         "uniform": {"shape", "stream"}, "integers": {"high", "shape", "stream"}, **{k: {"axis"} for k in REDUCE},
         **{k: {"axis"} for k in ALONG}}


class EntryError(ValueError):
    """The estimator is not a valid program, or it failed on a run. The message is for its author."""


# values ---------------------------------------------------------------------------------
def _norm(v) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype == np.bool_:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int64, copy=False)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64, copy=False)
    raise EntryError(f"values are integers, floats or booleans, not {a.dtype}")


def _guard(n: int) -> None:
    if n > MAX_ELEMENTS:
        raise EntryError(f"a value with {n} entries is above the limit of {MAX_ELEMENTS}")


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _axis(node: dict, ndim: int, key: str = "axis") -> int:
    ax = node.get(key)
    if not _whole(ax) or not 0 <= ax < max(ndim, 1) or ndim == 0:
        raise EntryError(f"{node['op']}: axis {ax!r} does not fit a value with {ndim} dimensions")
    return ax


def _int_index(v: np.ndarray, n: int, what: str) -> np.ndarray:
    if not np.issubdtype(v.dtype, np.integer):
        raise EntryError(f"{what}: an index must be whole numbers, not {v.dtype}")
    if v.size and (int(v.min()) < 0 or int(v.max()) >= n):
        raise EntryError(f"{what}: index out of range for a length of {n}")
    return v


def _const(node: dict) -> np.ndarray:
    dtype = node.get("dtype")
    if dtype not in DTYPES:
        raise EntryError(f"const: dtype must be one of {sorted(DTYPES)}")
    try:
        a = np.array(node.get("value"), dtype=DTYPES[dtype])
    except (ValueError, TypeError, OverflowError) as exc:
        raise EntryError(f"const: not a rectangular array of numbers ({exc})") from None
    if a.dtype == object or a.ndim > MAX_RANK:
        raise EntryError("const: not a rectangular array of numbers")
    if a.dtype == np.float64 and not np.isfinite(a).all():
        raise EntryError("const: numbers must be finite")
    return a


def _shape(node: dict) -> tuple:
    shape = node.get("shape")
    if not isinstance(shape, list) or len(shape) > MAX_RANK or any(not _whole(s) or s < 1 for s in shape):
        raise EntryError(f"{node['op']}: shape must be a list of at most {MAX_RANK} positive whole numbers")
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    _guard(size)
    return tuple(shape)


def svd_cost(x: np.ndarray) -> int:
    m, n = x.shape
    return SVD_FACTOR * m * n * min(m, n)


def _svd(x: np.ndarray):
    if x.ndim != 2 or 0 in x.shape:
        raise EntryError("svd: needs a matrix")
    u, s, vt = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    return (u, s, vt), svd_cost(x)


def apply(node: dict, args: list, weights: dict, rng=None, svd_store: dict | None = None):
    """The value and the cost of one node, given its arguments' values. cond is not handled here; a random
    node draws from rng."""
    op = node["op"]
    if op == "weight":
        name = node.get("name")
        if name not in weights:
            raise EntryError(f"weight: the network has no weights named {name!r}; it has {sorted(weights)}")
        return weights[name], 0
    if op == "const":
        return _const(node), 0
    if op == "arange":
        n = node.get("n")
        if not _whole(n) or not 0 < n <= MAX_ELEMENTS:
            raise EntryError("arange: n must be a positive whole number")
        return np.arange(n, dtype=np.int64), 0
    if op in RANDOM:
        shape = _shape(node)
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if rng is None:
            raise EntryError(f"{op}: no random draws are given here")
        if op == "normal":
            return _norm(rng.standard_normal(shape)), size
        if op == "uniform":
            return _norm(rng.random(shape)), size
        high = node.get("high")
        if not _whole(high) or not 0 < high <= 2**62:
            raise EntryError("integers: high must be a positive whole number")
        return _norm(rng.integers(0, high, size=shape, dtype=np.int64)), size

    if op in ELEMENTWISE_2:
        x, y = args
        shape = np.broadcast_shapes(x.shape, y.shape)
        size = int(np.prod(shape, dtype=np.int64))
        _guard(size)
        if op in ("and", "or") and not (x.dtype == np.bool_ and y.dtype == np.bool_):
            raise EntryError(f"{op}: needs booleans")
        if op in ("add", "sub", "mul", "div", "floordiv", "mod") and (x.dtype == np.bool_ or y.dtype == np.bool_):
            x, y = x.astype(np.int64), y.astype(np.int64)
        if op == "power":  # in floats: a whole number to a negative power is a fraction
            x, y = x.astype(np.float64), y.astype(np.float64)
        return _norm(ELEMENTWISE_2[op](x, y)), size
    if op in ELEMENTWISE_1:
        (x,) = args
        if op == "not" and x.dtype != np.bool_:
            raise EntryError("not: needs booleans")
        if op != "not" and x.dtype == np.bool_:
            x = x.astype(np.int64)
        if op in ("exp", "log", "sqrt", "erf"):
            x = x.astype(np.float64)
        return _norm(ELEMENTWISE_1[op](x)), int(x.size)
    if op == "where":
        c, x, y = args
        if c.dtype != np.bool_:
            raise EntryError("where: the condition must be boolean")
        shape = np.broadcast_shapes(c.shape, x.shape, y.shape)
        size = int(np.prod(shape, dtype=np.int64))
        _guard(size)
        return _norm(np.where(c, x, y)), size
    if op == "matmul":
        x, y = args
        if x.ndim == 0 or y.ndim == 0:
            raise EntryError("matmul: needs vectors or matrices; use mul for a scalar")
        k = x.shape[-1]
        if k != (y.shape[0] if y.ndim == 1 else y.shape[-2]):
            raise EntryError(f"matmul: shapes {x.shape} and {y.shape} do not fit")
        lead = np.broadcast_shapes(x.shape[:-2] if x.ndim > 1 else (), y.shape[:-2] if y.ndim > 1 else ())
        size = int(np.prod(lead, dtype=np.int64)) * (x.shape[-2] if x.ndim > 1 else 1) * (y.shape[-1] if y.ndim > 1 else 1)
        _guard(size)
        if size * max(k, 1) > MAX_WORK:
            raise EntryError(f"matmul: {size * k} multiply-adds are above the limit of {MAX_WORK} for one product")
        if x.dtype == np.bool_ or y.dtype == np.bool_:
            x, y = x.astype(np.int64), y.astype(np.int64)
        return _norm(np.matmul(x, y)), size * k
    if op in REDUCE:
        (x,) = args
        ax = node.get("axis")
        if ax is None:
            n, out = int(x.size), 1
        else:
            ax = _axis(node, x.ndim)
            n = x.shape[ax]
            out = int(x.size // n) if n else 0
        if n == 0:
            raise EntryError(f"{op}: nothing to reduce")
        if op in ("any", "all") and x.dtype != np.bool_:
            raise EntryError(f"{op}: needs booleans")
        return _norm(REDUCE[op](x, axis=ax)), out * (n - 1)
    if op in ALONG:
        (x,) = args
        ax = _axis(node, x.ndim)
        n = x.shape[ax]
        if n == 0:
            raise EntryError(f"{op}: an empty axis")
        vectors = int(x.size // n)
        if op == "cumsum":
            return _norm(np.cumsum(x.astype(np.int64) if x.dtype == np.bool_ else x, axis=ax)), vectors * (n - 1)
        if op == "softmax":
            z = x.astype(np.float64)
            z = np.exp(z - z.max(axis=ax, keepdims=True))
            return z / z.sum(axis=ax, keepdims=True), 3 * int(x.size)
        cost = vectors * n * math.ceil(math.log2(n)) if n > 1 else 0
        if op == "sort":
            return _norm(np.sort(x, axis=ax, kind="stable")), cost
        return _norm(np.argsort(x, axis=ax, kind="stable")), cost
    if op == "take":
        x, idx = args
        ax = _axis(node, x.ndim)
        _int_index(idx, x.shape[ax], "take")
        _guard(int(idx.size) * int(x.size // max(x.shape[ax], 1)))
        out = np.take(x, idx, axis=ax)
        return out, int(out.size)
    if op == "take_sum":
        x, idx = args
        ax = _axis(node, x.ndim)
        if idx.ndim != 1 or idx.size == 0:
            raise EntryError("take_sum: the index must be a non-empty vector")
        _int_index(idx, x.shape[ax], "take_sum")
        _guard(int(idx.size) * int(x.size // max(x.shape[ax], 1)))
        out = np.take(x, idx, axis=ax).sum(axis=ax)
        return _norm(out), int(out.size) * int(idx.size)
    if op == "gather":
        x, *idxs = args
        axes = node.get("axes")
        if (not isinstance(axes, list) or len(axes) != len(idxs) or not idxs or len(set(axes)) != len(axes)
                or any(not _whole(a) or not 0 <= a < x.ndim for a in axes)):
            raise EntryError("gather: axes must name one distinct axis of the value per index")
        index = [slice(None)] * x.ndim
        for a, i in zip(axes, idxs):
            if i.ndim != 0:
                raise EntryError("gather: every index must be a single whole number; use take for a vector of them")
            index[a] = int(_int_index(i, x.shape[a], "gather"))
        out = x[tuple(index)]
        return out, max(int(out.size), 1)
    if op in SVD:
        (x,) = args
        key = ("svd", node["args"][0])
        cost = 0
        if svd_store is None or key not in svd_store:
            parts, cost = _svd(x)
            if svd_store is not None:
                svd_store[key] = parts
        else:
            parts = svd_store[key]
        return parts[SVD.index(op)], cost
    if op == "reshape":
        (x,) = args
        shape = node.get("shape")
        if not isinstance(shape, list) or len(shape) > MAX_RANK or any(not _whole(s) or s < -1 for s in shape):
            raise EntryError("reshape: shape must be a list of sizes")
        try:
            return x.reshape(shape), 0
        except ValueError:
            raise EntryError(f"reshape: {x.shape} does not fit {shape}") from None
    if op == "transpose":
        (x,) = args
        perm = node.get("perm")
        if perm is not None and (not isinstance(perm, list) or sorted(perm) != list(range(x.ndim))):
            raise EntryError("transpose: perm must reorder the value's axes")
        return np.transpose(x, perm), 0
    if op == "slice":
        (x,) = args
        index = node.get("index")
        if not isinstance(index, list) or len(index) > x.ndim:
            raise EntryError("slice: one item per axis at most")
        idx = []
        for it in index:
            if it is None:
                idx.append(slice(None))
            elif _whole(it):
                idx.append(it)
            elif isinstance(it, dict) and set(it) <= {"start", "stop", "step"} and all(
                    v is None or _whole(v) for v in it.values()) and it.get("step") != 0:
                idx.append(slice(it.get("start"), it.get("stop"), it.get("step")))
            else:
                raise EntryError("slice: an item is null, a whole number or {start, stop, step}")
        try:
            return x[tuple(idx)], 0
        except IndexError:
            raise EntryError(f"slice: out of range for shape {x.shape}") from None
    if op in ("concat", "stack"):
        if not args:
            raise EntryError(f"{op}: needs at least one value")
        _guard(sum(int(a.size) for a in args))
        ax = node.get("axis")
        nd = args[0].ndim + (1 if op == "stack" else 0)
        if not _whole(ax) or not 0 <= ax < max(nd, 1) or nd > MAX_RANK:
            raise EntryError(f"{op}: axis {ax!r} does not fit")
        try:
            return _norm((np.concatenate if op == "concat" else np.stack)(args, axis=ax)), 0
        except ValueError:
            raise EntryError(f"{op}: shapes {[a.shape for a in args]} do not fit") from None
    if op == "cast":
        (x,) = args
        if node.get("dtype") not in DTYPES:
            raise EntryError(f"cast: dtype must be one of {sorted(DTYPES)}")
        if node["dtype"] == "int" and x.dtype == np.float64 and x.size and not (np.abs(x) < 2.0**63).all():
            # what a machine makes of such a float differs from one processor to another
            raise EntryError("cast: a float that is not finite, or too large, has no whole-number value")
        with np.errstate(all="ignore"):
            return x.astype(DTYPES[node["dtype"]]), int(x.size)
    raise EntryError(f"unknown operation {op!r}")


# estimators as data -----------------------------------------------------------------------
def check_structure(entry, max_nodes: int = MAX_NODES, max_const_scalars: int = MAX_CONST_SCALARS) -> dict:
    """Refuse anything that is not an estimator: the JSON shape, the operations, their arguments and attributes.
    Returns its size: {"nodes", "const_scalars", "random_nodes"}."""
    if not isinstance(entry, dict) or set(entry) != {"format", "nodes", "estimate"} or entry.get("format") != FORMAT:
        raise EntryError(f'an estimator is {{"format": "{FORMAT}", "nodes": [...], "estimate": n}}')
    nodes = entry.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise EntryError("an estimator needs a list of nodes")
    if len(nodes) > min(max_nodes, MAX_NODES):
        raise EntryError(f"the estimator has {len(nodes)} nodes, above the limit of {min(max_nodes, MAX_NODES)}")
    scalars, streams = 0, set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("op"), str) or node["op"] not in ARITY:
            raise EntryError(f"node {i}: unknown operation {node.get('op') if isinstance(node, dict) else node!r}")
        op = node["op"]
        extra = set(node) - {"op", "args"} - ATTRS.get(op, set())
        if extra:
            raise EntryError(f"node {i} ({op}): does not take {sorted(extra)}")
        args = node.get("args", [])
        if not isinstance(args, list) or any(not _whole(a) or not 0 <= a < i for a in args):
            raise EntryError(f"node {i} ({op}): arguments must be earlier nodes")
        if ARITY[op] is not None and len(args) != ARITY[op]:
            raise EntryError(f"node {i} ({op}): takes {ARITY[op]} arguments, not {len(args)}")
        if ARITY[op] is None and not 1 <= len(args) <= 64:
            raise EntryError(f"node {i} ({op}): takes between 1 and 64 arguments")
        if op == "const":
            scalars += int(_const(node).size)
            if scalars > max_const_scalars:
                raise EntryError(f"the estimator's constants hold more than the limit of {max_const_scalars} numbers")
        if op in RANDOM:
            _shape(node)
            s = node.get("stream")
            if not _whole(s) or not 0 <= s < MAX_STREAM:
                raise EntryError(f"node {i} ({op}): stream must be a whole number from 0 to 2^62")
            if s in streams:
                raise EntryError(f"node {i} ({op}): stream {s} is used by another random node; each draws from its own")
            streams.add(s)
    r = entry.get("estimate")
    if not _whole(r) or not 0 <= r < len(nodes):
        raise EntryError("estimate must name a node")
    return {"nodes": len(nodes), "const_scalars": scalars, "random_nodes": len(streams)}


def canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entry_hash(entry: dict) -> str:
    return hashlib.sha256(canonical(entry).encode()).hexdigest()


def show(entry: dict) -> str:
    """An estimator as readable lines, one per node: n3 = matmul(n0, n1)."""
    check_structure(entry)
    lines = []
    for i, node in enumerate(entry["nodes"]):
        op = node["op"]
        parts = [f"n{a}" for a in node.get("args", [])]
        for k in sorted(set(node) - {"op", "args"}):
            v = node[k]
            if op == "const" and k == "value":
                text = json.dumps(v)
                v = text if len(text) <= 60 else f"{text[:57]}..."
                parts.append(f"value={v}")
            else:
                parts.append(f"{k}={json.dumps(v)}")
        mark = "   <- estimate" if entry.get("estimate") == i else ""
        lines.append(f"n{i} = {op}({', '.join(parts)}){mark}")
    return "\n".join(lines)


def uses_randomness(entry: dict) -> bool:
    """Whether the estimate depends on a random draw (otherwise every run gives the same number at the same cost)."""
    check_structure(entry)
    dep = []
    for node in entry["nodes"]:
        dep.append(node["op"] in RANDOM or any(dep[a] for a in node.get("args", [])))
    return bool(dep[entry["estimate"]])


# the interpreter ----------------------------------------------------------------------------
class Evaluator:
    """Runs an estimator and counts its operations, one run at a time.

    A node that depends on no random draw (a fixed node) has the same value in every run, so it is computed
    the first time a run needs it and kept. Every run is charged for the fixed work it uses: the closure of a
    fixed node is the set of fixed nodes it took to compute it (for cond, the condition and the chosen side),
    and a run pays once for the union of the closures of the fixed nodes it reads. So a run costs what it
    would cost if it were the only one."""

    def __init__(self, entry: dict, weights: dict, deadline: float | None = None):
        self.size = check_structure(entry)
        self.nodes = entry["nodes"]
        self.root = entry["estimate"]
        self.weights = {k: _norm(v) for k, v in weights.items()}
        self.deadline = deadline
        self.dep = []
        for node in self.nodes:
            self.dep.append(node["op"] in RANDOM or any(self.dep[a] for a in node.get("args", [])))
        self.random = bool(self.dep[self.root])
        self.fixed: dict = {}  # node -> value, for the fixed nodes computed so far
        self.closure: dict = {}  # node -> frozenset of the fixed work it took (node ids, and ("svd", arg) keys)
        self.unit: dict = {}  # an element of a closure -> its own cost
        self._svd_fixed: dict = {}
        self._held_fixed = 0
        self._charged: dict = {}  # frozenset of fixed nodes a run read -> the fixed work that charges it

    def _have(self, j: int, memo: dict) -> bool:
        return j in memo if self.dep[j] else j in self.fixed

    def _val(self, j: int, memo: dict):
        return memo[j] if self.dep[j] else self.fixed[j]

    def run(self, seed) -> dict:
        """One run: {"estimate": the node's value (a numpy array), "ops": its operations, "draws": the random
        numbers it drew}. seed is a whole number or a list of them."""
        seed = [int(s) for s in (seed if isinstance(seed, (list, tuple)) else [seed])]
        if any(s < 0 for s in seed):
            raise EntryError("a seed is made of whole numbers that are not negative")
        memo: dict = {}
        read: set = set()  # the fixed nodes this run's random-dependent nodes read
        counter = [0, 0, 0]  # operations of the random-dependent nodes, random numbers drawn, entries held this run
        nodes, dep = self.nodes, self.dep
        stack = [self.root]
        while stack:
            i = stack[-1]
            if self._have(i, memo):
                stack.pop()
                continue
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise EntryError("the estimator ran out of time")
            node = nodes[i]
            op = node["op"]
            args = node.get("args", [])
            if op == "cond":
                c, a, b = args
                if not self._have(c, memo):
                    stack.append(c)
                    continue
                pred = self._val(c, memo)
                if pred.dtype != np.bool_ or pred.size != 1:
                    raise EntryError(f"node {i} (cond): the condition must be one boolean")
                chosen = a if bool(pred.reshape(())) else b
                if not self._have(chosen, memo):
                    stack.append(chosen)
                    continue
                value, cost, consumed = self._val(chosen, memo), 1, (c, chosen)
            else:
                missing = [j for j in args if not self._have(j, memo)]
                if missing:
                    stack.extend(reversed(missing))
                    continue
                vals = [self._val(j, memo) for j in args]
                rng = np.random.default_rng(seed + [int(node["stream"])]) if op in RANDOM else None
                try:
                    with np.errstate(all="ignore"):
                        value, cost = apply(node, vals, self.weights, rng, memo if dep[i] else self._svd_fixed)
                except EntryError as exc:
                    raise EntryError(f"node {i} ({op}): {exc}") from None
                except Exception as exc:  # whatever numpy made of it: the node failed, and its author is told which
                    raise EntryError(f"node {i} ({op}): {type(exc).__name__}: {str(exc)[:300]}") from None
                if value.size > MAX_ELEMENTS or value.ndim > MAX_RANK:
                    raise EntryError(f"node {i} ({op}): the value is too big")
                consumed = args
            if dep[i]:
                memo[i] = value
                counter[0] += int(cost)
                if op in RANDOM:
                    counter[1] += int(cost)
                counter[2] += int(value.size)
                for j in consumed:
                    if not dep[j]:
                        read.add(j)
            else:
                self.fixed[i] = value
                self._held_fixed += int(value.size)
                cl = {i}
                for j in consumed:
                    cl |= self.closure[j]
                if op in SVD:
                    key = ("svd", args[0])
                    self.unit[key] = svd_cost(self.fixed[args[0]])
                    self.unit[i] = 0
                    cl.add(key)
                else:
                    self.unit[i] = int(cost)
                self.closure[i] = frozenset(cl)
            if self._held_fixed + counter[2] > MAX_HELD:
                raise EntryError(f"the estimator holds more than {MAX_HELD} numbers at once")
            stack.pop()
        if not dep[self.root]:
            read.add(self.root)
        key = frozenset(read)
        if key not in self._charged:  # most runs read the same fixed nodes, so the union is made once
            work: set = set()
            for j in key:
                work |= self.closure[j]
            self._charged[key] = builtins.sum(self.unit[w] for w in work)
        ops = counter[0] + self._charged[key]
        return {"estimate": self._val(self.root, memo), "ops": int(ops), "draws": int(counter[1])}


def as_estimate(value) -> float:
    """The number an estimator's result stands for: one finite number, or the run is refused."""
    v = np.asarray(value)
    if v.size != 1 or not (np.issubdtype(v.dtype, np.number) or v.dtype == np.bool_):
        raise EntryError(f"the estimate must be one number, not a value of shape {v.shape} ({v.dtype})")
    x = float(v.reshape(()))
    if not math.isfinite(x):
        raise EntryError(f"the estimate is {x}, not a finite number")
    return x


def run_seed(master: int, k: int) -> list:
    """The seed of run k of a scoring whose seed is master."""
    return [int(master), int(k)]


def run_estimator(entry: dict, weights: dict, runs: int, master: int, deadline: float | None = None,
                  evaluator: Evaluator | None = None) -> dict:
    """The estimator run `runs` times, run k with seed run_seed(master, k): {"estimates", "ops", "draws"} (numpy
    arrays, one entry per run) and "random" (whether the estimate depends on a draw). An estimator that draws
    nothing gives the same number at the same cost every time, so it is run once and that run is repeated."""
    ev = evaluator or Evaluator(entry, weights, deadline)
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    n = runs if ev.random else 1
    est = np.empty(n, dtype=np.float64)
    ops = np.empty(n, dtype=np.int64)
    draws = np.empty(n, dtype=np.int64)
    for k in range(n):
        try:
            r = ev.run(run_seed(master, k))
            est[k] = as_estimate(r["estimate"])
        except EntryError as exc:
            raise EntryError(f"run {k} (seed [{int(master)}, {k}]): {exc}") from None
        ops[k], draws[k] = r["ops"], r["draws"]
    if n < runs:
        est, ops, draws = np.repeat(est, runs), np.repeat(ops, runs), np.repeat(draws, runs)
    return {"estimates": est, "ops": ops, "draws": draws, "random": ev.random}


# data sets -----------------------------------------------------------------------------------
# A model says what its data set is (config.yaml: data_set). Two kinds:
#   normal    an input is real numbers, one per position (the input space says `real` for each), each drawn
#             from the standard normal distribution on its own
#   all       an input is one token per position, each uniform over the position's size in the input space
KINDS = ("normal", "all")


def data_set_kind(data_set) -> str:
    kind = (data_set or {}).get("kind", "all")
    if kind not in KINDS:
        raise ValueError(f"a data set is of kind {' or '.join(KINDS)}, not {kind!r}")
    return kind


def draw_inputs(data_set, input_space, n: int, rng) -> np.ndarray:
    """n inputs drawn from the data set with a numpy generator, one row each (for exploring in plain numpy)."""
    if data_set_kind(data_set) == "normal":
        return rng.standard_normal((n, len(input_space)))
    return np.stack([rng.integers(0, int(k), size=n) for k in input_space], axis=1).astype(np.int64)


# the builder ---------------------------------------------------------------------------------
class Expr:
    """A node of a Graph being built. Its .value is the node's value in an example run (the graph's seed), so
    shapes are known and a mistake shows where it is made."""

    __slots__ = ("g", "id")
    __array_priority__ = 1000

    def __init__(self, g: "Graph", id: int):
        self.g, self.id = g, id

    @property
    def value(self) -> np.ndarray:
        return self.g.values[self.id]

    @property
    def shape(self) -> tuple:
        return self.value.shape

    @property
    def T(self) -> "Expr":
        return self.g.op("transpose", self, perm=None)

    def __repr__(self) -> str:
        return f"<Expr {self.id} {self.g.nodes[self.id]['op']} shape={self.shape}>"

    def __hash__(self) -> int:
        return hash((id(self.g), self.id))

    def __bool__(self):
        raise TypeError("an expression has no truth value while the estimator is being built; use where or cond")

    def _b(self, op, other, swap=False):
        return self.g.op(op, other, self) if swap else self.g.op(op, self, other)

    def __add__(self, o): return self._b("add", o)
    def __radd__(self, o): return self._b("add", o, True)
    def __sub__(self, o): return self._b("sub", o)
    def __rsub__(self, o): return self._b("sub", o, True)
    def __mul__(self, o): return self._b("mul", o)
    def __rmul__(self, o): return self._b("mul", o, True)
    def __truediv__(self, o): return self._b("div", o)
    def __rtruediv__(self, o): return self._b("div", o, True)
    def __floordiv__(self, o): return self._b("floordiv", o)
    def __rfloordiv__(self, o): return self._b("floordiv", o, True)
    def __mod__(self, o): return self._b("mod", o)
    def __rmod__(self, o): return self._b("mod", o, True)
    def __pow__(self, o): return self._b("power", o)
    def __rpow__(self, o): return self._b("power", o, True)
    def __matmul__(self, o): return self._b("matmul", o)
    def __rmatmul__(self, o): return self._b("matmul", o, True)
    def __lt__(self, o): return self._b("lt", o)
    def __le__(self, o): return self._b("le", o)
    def __gt__(self, o): return self._b("gt", o)
    def __ge__(self, o): return self._b("ge", o)
    def __eq__(self, o): return self._b("eq", o)
    def __ne__(self, o): return self._b("ne", o)
    def __and__(self, o): return self._b("and", o)
    def __rand__(self, o): return self._b("and", o, True)
    def __or__(self, o): return self._b("or", o)
    def __ror__(self, o): return self._b("or", o, True)
    def __invert__(self): return self.g.op("not", self)
    def __neg__(self): return self.g.op("neg", self)
    def __abs__(self): return self.g.op("abs", self)

    def reshape(self, *shape):
        shape = list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape)
        return self.g.op("reshape", self, shape=[int(s) for s in shape])

    def astype(self, dtype):
        return cast(self, dtype)

    def sum(self, axis=None): return _reduce("sum", self, axis)
    def max(self, axis=None): return _reduce("max", self, axis)
    def min(self, axis=None): return _reduce("min", self, axis)
    def argmax(self, axis=None): return _reduce("argmax", self, axis)
    def argmin(self, axis=None): return _reduce("argmin", self, axis)
    def any(self, axis=None): return _reduce("any", self, axis)
    def all(self, axis=None): return _reduce("all", self, axis)

    def mean(self, axis=None):
        """The sum over the axis (or all entries) divided by their number: a sum and a division."""
        n = self.value.size if axis is None else self.value.shape[_ax(self, axis)]
        return self.sum(axis) / float(n)

    def __getitem__(self, index):
        """x[...] with whole numbers, slices, None and expressions. Expressions that are one whole number read
        at those positions together (gather: 1 per entry read); one vector of whole numbers is allowed (take)."""
        items = list(index) if isinstance(index, tuple) else [index]
        if builtins.any(it is Ellipsis for it in items):
            raise TypeError("write the axes out; ... is not supported")
        has_new = builtins.any(it is None for it in items)
        if has_new and builtins.any(isinstance(it, Expr) for it in items):
            raise TypeError("None together with an expression index is not supported; reshape instead")
        static = []  # one item per indexed axis of this value: None (all of it), a whole number, or a slice
        for it in items:
            if it is None:
                continue
            if isinstance(it, Expr):
                static.append(None)
            elif isinstance(it, slice):
                spec = {}
                for k, v in (("start", it.start), ("stop", it.stop), ("step", it.step)):
                    if v is not None and (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))):
                        raise TypeError("slices take whole numbers")
                    spec[k] = None if v is None else int(v)
                static.append(None if builtins.all(v is None for v in spec.values()) else spec)
            elif isinstance(it, (int, np.integer)) and not isinstance(it, (bool, np.bool_)):
                static.append(int(it))
            else:
                raise TypeError(f"cannot index with {type(it).__name__}")
        if len(static) > self.value.ndim:
            raise IndexError(f"too many indices for a value of shape {self.shape}")
        cur = self
        if builtins.any(s is not None for s in static):
            cur = self.g.op("slice", self, index=static)
        pos, k = [], 0  # where each indexed axis sits after the static part (a whole number removed its axis)
        for s in static:
            if isinstance(s, int):
                pos.append(None)
            else:
                pos.append(k)
                k += 1
        indexed = [it for it in items if it is not None]
        dyn = [(pos[j], it) for j, it in enumerate(indexed) if isinstance(it, Expr)]
        scalars = [(a, e) for a, e in dyn if e.value.ndim == 0]
        vectors = [(a, e) for a, e in dyn if e.value.ndim != 0]
        if len(vectors) > 1:
            raise TypeError("at most one index may be a vector; use take for more")
        if scalars:
            cur = self.g.op("gather", cur, *[e for _, e in scalars], axes=[a for a, _ in scalars])
        if vectors:
            a, e = vectors[0]
            a -= builtins.sum(1 for sa, _ in scalars if sa < a)
            cur = self.g.op("take", cur, e, axis=a)
        if has_new:
            shape, dims, k = [], list(cur.value.shape), 0
            for it in items:
                if it is None:
                    shape.append(1)
                elif isinstance(it, slice):
                    shape.append(dims[k])
                    k += 1
            cur = cur.reshape(shape + dims[k:])
        return cur


class Graph:
    """Builds an estimator. Give it the network's weights, and the data set and its input space to draw inputs
    with g.inputs(n). Every expression is also computed in an example run with seed [seed], which is how shapes
    are known: a random node's example value is what it draws in that run."""

    def __init__(self, weights: dict, data_set: dict | None = None, input_space: list | None = None, seed: int = 0):
        self.weights = {k: _norm(v) for k, v in weights.items()}
        self.data_set = dict(data_set or {})
        self.input_space = list(input_space or [])
        self.seed = [int(seed)]
        self.nodes: list[dict] = []
        self.values: list[np.ndarray] = []
        self._seen: dict[str, int] = {}
        self._svd: dict = {}
        self._stream = 0

    @property
    def n_ctx(self) -> int:
        """Numbers (or tokens) per input of the data set."""
        return len(self.input_space)

    # sources
    def weight(self, name: str) -> Expr:
        return self._add({"op": "weight", "name": str(name)}, [])

    def arange(self, n: int) -> Expr:
        return self._add({"op": "arange", "n": int(n)}, [])

    def const(self, value) -> Expr:
        a = np.asarray(value)
        if a.dtype == np.bool_:
            dtype = "bool"
        elif np.issubdtype(a.dtype, np.integer):
            dtype = "int"
        elif np.issubdtype(a.dtype, np.floating):
            dtype = "float"
        else:
            raise TypeError(f"a constant is a number or an array of numbers, not {a.dtype}")
        return self._add({"op": "const", "value": a.tolist(), "dtype": dtype}, [])

    def _random(self, op: str, shape, **attrs) -> Expr:
        shape = [int(s) for s in (shape if isinstance(shape, (list, tuple)) else [shape])]
        node = {"op": op, "shape": shape, "stream": self._stream, **attrs}
        self._stream += 1
        return self._add(node, [])

    def normal(self, shape) -> Expr:
        """Numbers drawn from the standard normal distribution, fresh in every run: 1 per number."""
        return self._random("normal", shape)

    def uniform(self, shape) -> Expr:
        """Numbers drawn uniformly from [0, 1), fresh in every run: 1 per number."""
        return self._random("uniform", shape)

    def integers(self, high: int, shape) -> Expr:
        """Whole numbers drawn uniformly from 0 to high - 1, fresh in every run: 1 per number."""
        return self._random("integers", shape, high=int(high))

    def inputs(self, n: int) -> Expr:
        """n inputs drawn from the data set, fresh in every run, one row each: (n, numbers per input), at 1 per
        number drawn."""
        if not self.input_space:
            raise ValueError("this graph was made without the data set; make it with graph()")
        if data_set_kind(self.data_set) == "normal":
            return self.normal([int(n), self.n_ctx])
        cols = [self.integers(int(k), [int(n), 1]) for k in self.input_space]
        return concat(cols, axis=1)

    def wrap(self, x) -> Expr:
        if isinstance(x, Expr):
            if x.g is not self:
                raise ValueError("expressions of two graphs cannot be mixed")
            return x
        return self.const(x)

    def op(self, name: str, *args, **attrs) -> Expr:
        args = [self.wrap(a) for a in args]
        node = {"op": name, "args": [a.id for a in args], **{k: v for k, v in attrs.items() if v is not None or k == "axis"}}
        if name in REDUCE and node.get("axis") is None:
            node["axis"] = None
        return self._add(node, args)

    def _add(self, node: dict, args: list) -> Expr:
        if "args" in node and not node["args"] and node["op"] in SOURCES + RANDOM:
            node = {k: v for k, v in node.items() if k != "args"}
        key = json.dumps(node, sort_keys=True)
        if key in self._seen:
            return Expr(self, self._seen[key])
        vals = [a.value for a in args]
        with np.errstate(all="ignore"):
            if node["op"] == "cond":
                c = vals[0]
                if c.dtype != np.bool_ or c.size != 1:
                    raise EntryError("cond: the condition must be one boolean")
                value = vals[1] if bool(c.reshape(())) else vals[2]
            else:
                rng = np.random.default_rng(self.seed + [int(node["stream"])]) if node["op"] in RANDOM else None
                value, _ = apply(node, vals, self.weights, rng, self._svd)
        if node["op"] in RANDOM:
            self._stream = max(self._stream, int(node["stream"]) + 1)
        self.nodes.append(node)
        self.values.append(value)
        self._seen[key] = len(self.nodes) - 1
        return Expr(self, len(self.nodes) - 1)

    def entry(self, estimate) -> dict:
        """The estimator as JSON data: the nodes the estimate needs, renumbered."""
        estimate = self.wrap(estimate)
        need, stack = set(), [estimate.id]
        while stack:
            i = stack.pop()
            if i not in need:
                need.add(i)
                stack.extend(self.nodes[i].get("args", []))
        order = sorted(need)
        new = {old: k for k, old in enumerate(order)}
        nodes = []
        for old in order:
            node = dict(self.nodes[old])
            if "args" in node:
                node["args"] = [new[a] for a in node["args"]]
            nodes.append(node)
        entry = {"format": FORMAT, "nodes": nodes, "estimate": new[estimate.id]}
        check_structure(entry)
        return entry

    @classmethod
    def load(cls, entry: dict, weights: dict, data_set: dict | None = None, input_space: list | None = None,
             seed: int = 0) -> tuple:
        """An estimator opened for more building: (graph, estimate). New random nodes get streams of their own."""
        check_structure(entry)
        g = cls(weights, data_set, input_space, seed)
        exprs = []
        for node in entry["nodes"]:
            node = dict(node)
            args = [exprs[a] for a in node.get("args", [])]
            if "args" in node:
                node["args"] = [a.id for a in args]
            exprs.append(g._add(node, args))
        return g, exprs[entry["estimate"]]


def _g(*xs) -> Graph:
    for x in xs:
        if isinstance(x, Expr):
            return x.g
        if isinstance(x, (list, tuple)):
            for y in x:
                if isinstance(y, Expr):
                    return y.g
    raise TypeError("at least one argument must be an expression of a graph")


def _ax(x: Expr, axis) -> int:
    nd = x.value.ndim
    axis = int(axis)
    if not -nd <= axis < nd:
        raise ValueError(f"axis {axis} does not fit a value with {nd} dimensions")
    return axis % nd


def _reduce(op: str, x, axis) -> Expr:
    g = _g(x)
    x = g.wrap(x)
    return g.op(op, x, axis=None if axis is None else _ax(x, axis))


def where(c, x, y) -> Expr: return _g(c, x, y).op("where", c, x, y)
def cond(c, x, y) -> Expr: return _g(c, x, y).op("cond", c, x, y)
def matmul(x, y) -> Expr: return _g(x, y).op("matmul", x, y)
def maximum(x, y) -> Expr: return _g(x, y).op("maximum", x, y)
def minimum(x, y) -> Expr: return _g(x, y).op("minimum", x, y)
def power(x, y) -> Expr: return _g(x, y).op("power", x, y)
def exp(x) -> Expr: return _g(x).op("exp", x)
def log(x) -> Expr: return _g(x).op("log", x)
def sqrt(x) -> Expr: return _g(x).op("sqrt", x)
def erf(x) -> Expr: return _g(x).op("erf", x)
def argmax(x, axis=None) -> Expr: return _reduce("argmax", x, axis)
def argmin(x, axis=None) -> Expr: return _reduce("argmin", x, axis)


def normal_cdf(x) -> Expr:
    """P(Z <= x) for a standard normal Z, as 0.5 * (1 + erf(x / sqrt(2))): three operations per entry."""
    return 0.5 * (1.0 + erf(x * (1.0 / math.sqrt(2.0))))


def _along(op: str, x, axis) -> Expr:
    g = _g(x)
    x = g.wrap(x)
    return g.op(op, x, axis=_ax(x, axis))


def cumsum(x, axis=-1) -> Expr: return _along("cumsum", x, axis)
def sort(x, axis=-1) -> Expr: return _along("sort", x, axis)
def argsort(x, axis=-1) -> Expr: return _along("argsort", x, axis)
def softmax(x, axis=-1) -> Expr: return _along("softmax", x, axis)


def take(x, idx, axis=0) -> Expr:
    g = _g(x, idx)
    x = g.wrap(x)
    return g.op("take", x, idx, axis=_ax(x, axis))


def take_sum(x, idx, axis=0) -> Expr:
    g = _g(x, idx)
    x = g.wrap(x)
    return g.op("take_sum", x, idx, axis=_ax(x, axis))


def gather(x, idxs, axes) -> Expr:
    g = _g(x, idxs)
    x = g.wrap(x)
    return g.op("gather", x, *idxs, axes=[_ax(x, a) for a in axes])


def concat(xs, axis=0) -> Expr:
    g = _g(xs)
    xs = [g.wrap(x) for x in xs]
    return g.op("concat", *xs, axis=_ax(xs[0], axis))


def stack(xs, axis=0) -> Expr:
    g = _g(xs)
    xs = [g.wrap(x) for x in xs]
    nd = xs[0].value.ndim + 1
    return g.op("stack", *xs, axis=int(axis) % nd)


def reshape(x, shape) -> Expr: return _g(x).wrap(x).reshape(shape)
def transpose(x, perm=None) -> Expr: return _g(x).op("transpose", x, perm=None if perm is None else [int(p) for p in perm])


def cast(x, dtype) -> Expr:
    name = {int: "int", float: "float", bool: "bool"}.get(dtype, dtype)
    return _g(x).op("cast", x, dtype=name)


def svd(x) -> tuple:
    """(u, s, vt) of a matrix, as numpy's reduced decomposition. Charged once for the three."""
    g = _g(x)
    return g.op("svd_u", x), g.op("svd_s", x), g.op("svd_vt", x)
