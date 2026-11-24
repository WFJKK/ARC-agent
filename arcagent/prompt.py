"""A round's prompt: the template, the network's and the task's definitions, the estimator guide, the scoring and
the history.

Everything the agent is told goes through here, and through hygiene.check before it is sent. The prompt never
names a network, a model folder or anything of the pipeline: the agent has "the network" and "other networks".
"""

from __future__ import annotations

from . import rewards
from .util import REPO_ROOT

TEMPLATES = REPO_ROOT / "prompts"
ESTIMATOR_DOC = REPO_ROOT / "agent_docs" / "ESTIMATOR.md"
COPIES_DOC = REPO_ROOT / "agent_docs" / "COPIES.md"  # the part of the guide for a run that scores on copies


def _e(v, digits=4) -> str:
    return "none" if v is None else f"{v:.{digits}e}"


def _f(v, digits=4) -> str:
    return "none" if v is None else f"{v:.{digits}f}"


def history(rounds: list) -> str:
    """Earlier rounds as the agent sees them: every submission's result in a table, then the round's notes."""
    if not rounds:
        return "None: this is the first round."
    out = []
    best = None
    for rd in rounds:
        out.append(f"### Round {rd['round']}")
        subs = rd.get("submissions") or []
        if subs:
            out.append("")
            # a scoring on copies gives the spread of the estimates twice: on one copy, and from copy to copy
            copies = any("estimate_std_between_copies" in (s.get("seen") or {}) for s in subs)
            head = ["saved as", "status", "reward", "mean squared error", "cost (forward passes)", "estimate: mean"]
            head += ["estimate: spread on a copy", "estimate: spread between copies"] if copies else ["estimate: spread"]
            head.append("draws")
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + " --- |" * len(head))
            for s in subs:
                v = s.get("seen") or {}
                if s["status"] == "scored":
                    row = [s["carried_as"], "scored", _e(s["reward"]), _e(v.get("mean_squared_error")), _f(v.get("cost"), 2),
                           _f(v.get("estimate_mean"), 5), _e(v.get("estimate_std"), 2)]
                    if copies:
                        row.append(_e(v.get("estimate_std_between_copies"), 2))
                    row.append("yes" if v.get("uses_random_draws") else "no")
                    if best is None or s["reward"] > best[1]:
                        best = (s["carried_as"], s["reward"])
                else:
                    row = [s.get("name") or "", f"refused: {str(s.get('reason'))[:120]}"] + [""] * (len(head) - 2)
                out.append("| " + " | ".join(row) + " |")
        else:
            out.append("")
            out.append("No submission.")
        out.append("")
        out.append("Notes of that round:")
        out.append("")
        notes = (rd.get("notes") or "").strip() or "(none)"
        out.append("\n".join("    " + line for line in notes.splitlines()))
        out.append("")
    if best is not None:
        out.append(f"The scored estimators above are in `estimators` under the names in the first column, and the best so "
                   f"far (reward {_e(best[1])}) is also there as `best`.")
    return "\n".join(out)


RESTART_NOTE = """

## This session was started before

It was cut off, and it starts again here with the same task and what is left of its budgets:
{python_left} of {python_all} `run_python` calls and {submit_left} of {submit_all} submissions. The
saved estimators are still in `estimators`: {saved}.
{submitted}"""


def restart_note(session) -> str:
    """What a session that was cut off is told when it starts again: its budgets as they stand, the estimators it
    saved and the submissions it already made. Empty for a session that has used nothing."""
    used, budgets = session.state["used"], session.state["budgets"]
    if not any(used.values()):
        return ""
    lines = []
    for r in session.receipts():
        p = r.get("parts") or {}
        if r["status"] == "scored":
            lines.append(f"- `{r.get('name')}`: reward {_e(r['reward'])}, mean squared error {_e(p.get('mse'))}, "
                         f"cost {_f(p.get('cost'), 2)} forward passes")
        else:
            lines.append(f"- `{r.get('name')}`: refused: {str(r.get('reason'))[:200]}")
    submitted = ("It already submitted:\n\n" + "\n".join(lines) + "\n") if lines else "It had submitted nothing yet.\n"
    saved = ", ".join(f"`{n}`" for n in sorted(session.estimators())) or "none"
    return RESTART_NOTE.format(python_left=budgets["run_python"] - used["run_python"], python_all=budgets["run_python"],
                               submit_left=budgets["submit_estimator"] - used["submit_estimator"],
                               submit_all=budgets["submit_estimator"], saved=saved, submitted=submitted)


def estimator_doc(model, copies: bool = False) -> str:
    """The estimator guide (agent_docs/ESTIMATOR.md) for this model: its example reads the model's first weight, and
    its price table gives what drawing one of the data set's inputs costs. For a run that scores on copies of the
    network it has the part on copies (agent_docs/COPIES.md) too."""
    name = next(iter(model.config["weights"]))
    example = f'W = g.weight("{name}")'
    text = ESTIMATOR_DOC.read_text().strip()
    return (text.replace('W = g.weight("{{A_WEIGHT}}"){{PAD}}', example + " " * max(1, 35 - len(example)))
            .replace("{{INPUT_OPS}}", f"{model.input_ops():,}")
            .replace("{{COPIES_DOC}}", COPIES_DOC.read_text() if copies else ""))


def reference_inputs(n: int, model) -> str:
    """The inputs the verifier measures an accuracy on, in words: all of them when the data set has no more than it
    would draw (measure.accuracy)."""
    if model.n_total is not None and model.n_total <= int(n):
        return f"every one of the data set's {model.n_total:,} inputs, each weighted by how often it is drawn,"
    return f"{int(n):,} inputs it draws from the data set"


def _number(x: float) -> str:
    return f"{x:.4g}"


def scoring(cfg: dict, model, forward_ops: int) -> dict:
    """The fields of the prompt that say how an estimator is scored: on the network itself, or on copies of it
    (verifier.copies in the run config). Both texts are files in prompts/."""
    from . import measure

    ver = cfg["verifier"]
    spec = measure.copies_spec(ver.get("copies"))
    if spec is None:
        text = (TEMPLATES / "scoring_network_v1.md").read_text()
        fields = {"REFERENCE_INPUTS": reference_inputs(ver["reference_inputs"], model), "RUNS": f"{int(ver['runs']):,}"}
        note = ""
    else:
        text = (TEMPLATES / "scoring_copies_v1.md").read_text()
        fields = {"COPIES": f"{spec['count']:,}", "COPY_RUNS": f"{spec['runs']:,}", "RUNS": f"{spec['count'] * spec['runs']:,}",
                  "COPY_SHARE": _number(spec["share"]), "COPY_SIZE": _number(spec["size"]),
                  "COPY_REFERENCE": reference_inputs(spec["reference_inputs"], model)}
        note = "\n\n" + (TEMPLATES / "scored_on_copies_v1.md").read_text().strip()
    fields["MODEL_OPS"] = f"{forward_ops:,}"
    for k, v in fields.items():
        text = text.replace("{{" + k + "}}", str(v))
    return {"SCORING": text.strip(), "SCORED_ON": note, "RUNS": fields["RUNS"], "COPIES": spec is not None}


def build(cfg: dict, model, round_no: int, rounds: list, forward_ops: int) -> str:
    template = (TEMPLATES / f"{cfg['prompt_template']}.md").read_text()
    sizes = ", ".join(f"{k} = {v}" for k, v in model.sizes.items()) + f", outputs: {model.config['n_outputs']}"
    how = scoring(cfg, model, forward_ops)
    fields = {
        "ROUND": round_no, "ROUNDS_MAX": cfg["rounds_max"], "MODEL_SOURCE": model.source().strip(), "SIZES": sizes,
        "TASK": model.task_text(), "TASK_SOURCE": model.task_source().strip(), "DATA_SET": model.data_set_text(),
        "MODEL_OPS": f"{forward_ops:,}", "ESTIMATOR_DOC": estimator_doc(model, how["COPIES"]),
        "SCORING": how["SCORING"], "SCORED_ON": how["SCORED_ON"], "RUNS": how["RUNS"],
        "REWARD_FORMULA": rewards.formula(cfg["reward"]), "REWARD_EXPLANATION": rewards.explanation(cfg["reward"]),
        "PYTHON_CALLS": cfg["session"]["python_calls"], "PYTHON_SECONDS": cfg["python"]["time_limit_s"],
        "SUBMISSIONS": cfg["session"]["submissions"], "MAX_NODES": cfg["entry"]["max_nodes"],
        "MAX_CONST": cfg["entry"]["max_const_scalars"], "VERIFIER_SECONDS": f"{float(cfg['verifier']['time_limit_s']):g}",
        "HISTORY": history(rounds),
    }
    text = template
    for k, v in fields.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text:
        raise ValueError(f"the prompt template has a field nobody fills: {text[text.index('{{'):][:40]}")
    return text
