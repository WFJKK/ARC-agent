"""Make rnn-g's noised networks: copies of the first network, rnn-g-s0, with noise added to every weight.

    python withheld/rnn-g/noise.py      writes the networks that are missing, checks the others

Network rnn-g-n<k> is rnn-g-s0 with sigma_k times the matrix's root mean square times a standard normal draw
added to every weight of each matrix, in float64, then stored in float32 as the zoo stores its networks. The
draws come from numpy.random.default_rng([2026, k]), one matrix after another in the order input, recurrent,
readout. The network is very sensitive to its weights: noise of 0.2% of the RMS already costs about a point of
accuracy (task.yaml). A noised copy keeps the structure of the first network and moves its accuracy, which is
what tells an estimator that computes from the weights from a number written into it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZOO = HERE.parent.parent / "models" / "rnn-g" / "zoo"
SIGMAS = {1: 0.002, 2: 0.005, 3: 0.01}
ORDER = ("input", "recurrent", "readout")


def noised(weights: dict, k: int) -> dict:
    rng = np.random.default_rng([2026, k])
    out = {}
    for name in ORDER:
        w = weights[name].astype(np.float64)
        rms = float(np.sqrt((w**2).mean()))
        out[name] = (w + SIGMAS[k] * rms * rng.standard_normal(w.shape)).astype(np.float32)
    return out


def main() -> None:
    with np.load(ZOO / "rnn-g-s0.npz", allow_pickle=False) as z:
        first = {k: z[k] for k in z.files}
    for k in SIGMAS:
        w = noised(first, k)
        out = ZOO / f"rnn-g-n{k}.npz"
        if out.exists():
            with np.load(out, allow_pickle=False) as z:
                if set(z.files) != set(w) or not all(np.array_equal(z[n], w[n]) for n in w):
                    raise SystemExit(f"{out.name} differs from what this script makes; it is left as it is")
            print(f"{out.name} checked")
        else:
            np.savez(out, **w)
            print(f"{out.name} written")


if __name__ == "__main__":
    main()
