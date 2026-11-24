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
            out.append("| saved as | status | reward | mean squared error | cost (forward passes) | estimate: mean | estimate: spread | draws |")
            out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for s in subs:
                v = s.get("seen") or {}
                if s["status"] == "scored":
                    row = [s["carried_as"], "scored", _e(s["reward"]), _e(v.get("mean_squared_error")), _f(v.get("cost"), 2),
                           _f(v.get("estimate_mean"), 5), _e(v.get("estimate_std"), 2),
                           "yes" if v.get("uses_random_draws") else "no"]
                    if best is None or s["reward"] > best[1]:
                        best = (s["carried_as"], s["reward"])
                else:
                    row = [s.get("name") or "", f"refused: {str(s.get('reason'))[:120]}", "", "", "", "", "", ""]
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


def build(cfg: dict, model, round_no: int, rounds: list, forward_ops: int) -> str:
    template = (TEMPLATES / f"{cfg['prompt_template']}.md").read_text()
    sizes = ", ".join(f"{k} = {v}" for k, v in model.sizes.items()) + f", outputs: {model.config['n_outputs']}"
    fields = {
        "ROUND": round_no, "ROUNDS_MAX": cfg["rounds_max"], "MODEL_SOURCE": model.source().strip(), "SIZES": sizes,
        "TASK": model.task_text(), "TASK_SOURCE": model.task_source().strip(), "DATA_SET": model.data_set_text(),
        "MODEL_OPS": f"{forward_ops:,}", "ESTIMATOR_DOC": ESTIMATOR_DOC.read_text().strip(),
        "REFERENCE_INPUTS": f"{int(cfg['verifier']['reference_inputs']):,}", "RUNS": f"{int(cfg['verifier']['runs']):,}",
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
