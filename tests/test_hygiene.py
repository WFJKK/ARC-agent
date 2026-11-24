"""What the agent can reach holds none of what is withheld: the networks' accuracies, which network is which, and
where the networks come from (how they were made, which inputs they were trained on, the names that lead to
published figures). Checked for every pilot config, whatever its model."""

import pytest

from arcagent import hygiene, loop, prompt, run_config
from arcagent.networks import load_model
from arcagent.util import REPO_ROOT, read_yaml

PILOTS = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "config" / "pilot").glob("*.yaml")
                if p.name != "base.yaml")
RECORDS = sorted(p.parent.name for p in (REPO_ROOT / "withheld").glob("*/task.yaml"))


def accuracy_forms(a: float) -> list:
    """The ways an accuracy would be written: four places and two of percent, and where they do not round to 1,
    three places and one of percent."""
    out = [f"{a:.4f}", f"{100 * a:.2f}%"]
    return out + ([f"{a:.3f}", f"{100 * a:.1f}%"] if a < 0.9995 else [])


def test_every_model_has_a_pilot_and_a_withheld_record():
    models = sorted(p.parent.name for p in (REPO_ROOT / "models").glob("*/config.yaml"))
    assert RECORDS == models and len(models) == 20
    piloted = {run_config.load(c, {"runs_dir": "/nonexistent"})["model"].split("/")[-1] for c in PILOTS}
    assert piloted == set(models)


@pytest.mark.parametrize("config_file", PILOTS)
def test_the_first_prompt_and_everything_else_the_agent_reaches_are_clean(config_file, tmp_path):
    cfg = run_config.load(config_file, {"runs_dir": str(tmp_path)})
    model = load_model(cfg["model"])
    F = model.forward_ops(model.load_weights(cfg["network"]))
    texts = {"prompt": prompt.build(cfg, model, 1, [], F), **loop.agent_surface(cfg, model, F, "process")}
    forbidden = loop.forbidden_strings(cfg, model)
    assert len(forbidden) > 10 and not hygiene.hits(texts, forbidden)
    for net in model.network_ids():
        assert all(net not in t for t in texts.values())  # no network is named
    assert all("train" not in t.lower() for t in texts.values())  # nothing is said of training, or of none
    told = " ".join(texts["prompt"].split())
    assert model.task_text() in told and model.data_set_text() in told and "sandbox file task.py" in texts


@pytest.mark.parametrize("name", RECORDS)
def test_every_recorded_accuracy_is_on_the_hygiene_list(name):
    rec = read_yaml(REPO_ROOT / "withheld" / name / "task.yaml")
    words = [str(w) for w in rec["hygiene"]]
    for net, a in rec["accuracy"].items():
        for form in accuracy_forms(a):
            assert hygiene.hits({"text": f"about {form} of the inputs"}, words), (net, form)
    origin = {"AlgZoo", "alignment.org", "arc-ml"} if name.startswith("rnn") else {"Gross", "compact proof"}
    assert origin <= set(words)
    if "untrained" in " ".join(rec["networks"].split()).lower():
        assert {"untrained", "baseline", "initialization"} <= set(words)


def test_a_hit_stops_the_run_whatever_the_letter_case():
    found = hygiene.hits({"a": "line one\nthe ALGZOO network", "b": "nothing"}, ["algzoo"])
    assert found == [("a", "algzoo", "the ALGZOO network")]
    with pytest.raises(hygiene.HygieneError, match="withheld from it"):
        hygiene.check({"a": "measured 0.953 here"}, ["0.953"])
