"""What the agent can reach holds none of what is withheld: the networks' accuracies, which network is which, and the
names that lead to published figures."""

import pytest

from arcagent import hygiene, loop, prompt, run_config
from arcagent.networks import load_model
from arcagent.util import REPO_ROOT, read_yaml


@pytest.mark.parametrize("config_file", ["config/pilot/rnn-g.yaml", "config/pilot/rnn-b-smoke.yaml"])
def test_the_first_prompt_and_everything_else_the_agent_reaches_are_clean(config_file, tmp_path):
    cfg = run_config.load(config_file, {"runs_dir": str(tmp_path)})
    model = load_model(cfg["model"])
    F = model.forward_ops(model.load_weights(cfg["network"]))
    texts = {"prompt": prompt.build(cfg, model, 1, [], F), **loop.agent_surface(cfg, model, F, "process")}
    forbidden = loop.forbidden_strings(cfg, model)
    assert len(forbidden) > 10 and not hygiene.hits(texts, forbidden)
    for net in model.network_ids():
        assert all(net not in t for t in texts.values())  # no network is named
    assert "second-largest" in texts["prompt"] and "sandbox file task.py" in texts  # the task is told


@pytest.mark.parametrize("name", ["rnn-g", "rnn-b"])
def test_every_recorded_accuracy_is_on_the_hygiene_list(name):
    rec = read_yaml(REPO_ROOT / "withheld" / name / "task.yaml")
    words = [str(w) for w in rec["hygiene"]]
    for net, a in rec["accuracy"].items():
        assert f"{a:.3f}" in words, (net, a)
        assert f"{100 * a:.1f}%" in words, (net, a)
    assert {"AlgZoo", "alignment.org", "arc-ml"} <= set(words)


def test_a_hit_stops_the_run_whatever_the_letter_case():
    found = hygiene.hits({"a": "line one\nthe ALGZOO network", "b": "nothing"}, ["algzoo"])
    assert found == [("a", "algzoo", "the ALGZOO network")]
    with pytest.raises(hygiene.HygieneError, match="withheld from it"):
        hygiene.check({"a": "measured 0.953 here"}, ["0.953"])
