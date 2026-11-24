"""Reference estimators for rnn-a, for reports and tests. Withheld from the agent: no tool reaches this folder.

Each function takes a graph g, the estimator language lang, and the model's and the task's modules, and returns the
estimate as an expression.

Two facts about the network make its accuracy a short computation.

- It has no biases, and a ReLU commutes with a positive factor: an input scaled by c > 0 gives the logits scaled by
  c, so the output depends only on the input's direction. The task's answer (the position of the smaller number)
  does too, and the two numbers are independent standard normal draws, whose direction is uniform on the circle.
  The accuracy is the share of the circle on which the output is the answer.
- Along a straight line of inputs the logits are linear between the points at which a unit switches on or off.
  The four sides of the square with corners (+-1, +-1) are such lines and meet every direction once: the side
  (c, t) or (t, c), c = +-1, -1 <= t <= 1, covers the angles atan(t) of a quarter of the circle.

So the estimator finds, on each side, the points at which a unit switches (the zeros of a few linear expressions in
t), runs the network at those points and the side's ends, and between two neighbouring points knows the sign of
logit 0 less logit 1 exactly: it is linear there. It costs a forward pass per point, 22 in all, and its only
error is the polynomial that stands for atan (below 1e-5). Where no unit is on, both logits are 0 and the network
outputs 0 (the first on a tie): the estimator counts that stretch as output 0 too.
"""

import math

# atan(t) for -1 <= t <= 1, as t times a polynomial in t^2 (Abramowitz and Stegun 4.4.47; error below 1e-5)
ATAN = (0.9998660, -0.3302995, 0.1801410, -0.0851330, 0.0208351)


def _atan(t):
    u = t * t
    return t * (ATAN[0] + u * (ATAN[1] + u * (ATAN[2] + u * (ATAN[3] + u * ATAN[4]))))


def regions(g, lang, model, task):
    w = g.weight("input")[:, 0]          # one weight per unit
    rec = g.weight("recurrent")
    d = w.shape[0]

    def clip(x, lo, hi):
        return lang.minimum(lang.maximum(x, lo), hi)

    def zero_of(slope, at_zero, lo, hi):
        """Where slope * t + at_zero = 0, kept inside [lo, hi] (an end if it lies outside, or if there is no slope)."""
        t = -at_zero / lang.where(slope == 0.0, 1.0, slope)
        return clip(lang.where(slope == 0.0, hi, t), lo, hi)

    total = 0.0
    for first_fixed in (True, False):
        for c in (1.0, -1.0):
            if first_fixed:
                # inputs (c, t): the state after the first number is fixed, each unit's input after the second is
                # w t + (rec h1)
                h1 = lang.maximum(w * c, 0.0)
                points = lang.concat([g.const([-1.0, 1.0]), zero_of(w, rec @ h1, -1.0, 1.0)], axis=0)
                answer_first = c < 0           # c = -1 is the smaller number, at position 0
            else:
                # inputs (t, c): the state after the first number is t relu(w) for t > 0 and -t relu(-w) for t < 0;
                # each unit's input after the second number is w c + (rec of that) and is linear on either half
                up, down = rec @ lang.maximum(w, 0.0), rec @ lang.maximum(-w, 0.0)
                points = lang.concat([g.const([-1.0, 0.0, 1.0]), zero_of(up, w * c, 0.0, 1.0), zero_of(-down, w * c, -1.0, 0.0)],
                                     axis=0)
                answer_first = c > 0           # t < 1 = c: the first number is the smaller
            t = lang.sort(points, axis=0)
            fixed = g.const([c] * (points.shape[0]))
            x = lang.stack([fixed, t] if first_fixed else [t, fixed], axis=1)
            logits = model.logits(g, lang, x)
            diff = logits[:, 0] - logits[:, 1]            # the output is 0 where this is not negative
            # Where no unit is on, both logits are 0 and the output is 0. Such a stretch ends at a point where a unit
            # switches on, and there the arithmetic leaves a difference of 1e-16 in place of 0, with either sign:
            # what is that small against the side's largest difference is 0.
            diff = lang.where(abs(diff) <= 1e-9 * abs(diff).max(), 0.0, diff)
            lo, hi, a, b = t[:-1], t[1:], diff[:-1], diff[1:]
            angle = _atan(t)
            width = angle[1:] - angle[:-1]
            # between two neighbouring points diff is linear: where it changes sign, it does so at `cross`
            cross = _atan(lo + (hi - lo) * a / lang.where(a == b, 1.0, a - b))
            zero = lang.where((a >= 0) & (b >= 0), width,
                              lang.where((a < 0) & (b < 0), 0.0, lang.where(a >= 0, cross - angle[:-1], angle[1:] - cross)))
            total = total + (zero.sum() if answer_first else width.sum() - zero.sum())
    return total / (2.0 * math.pi)


ESTIMATORS = {
    "the regions on which the logits are linear, exactly": regions,
}
