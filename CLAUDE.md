# Working rules

This repository is the ARC-agent pipeline described in `docs/DESIGN.md`: an agent writes an estimator of a small
network's accuracy in a language of priced operations, and a verifier measures its error and its compute. It is
built on Mech-Interp-Agent-level1, and keeps its rules.

- The agent is told the network, the task and the data set, and never the answer: how often a network gives the
  task's answer, which network is which, or where the networks come from. This holds for every route: the prompt,
  the tools' descriptions, the files placed in its sandbox, names (model folders, network ids, weights), the
  estimator guide, examples, and what its code can import. Accuracies and provenance are said in `withheld/`, in
  the tests and in the README, and nowhere else in the repository. `arcagent/hygiene.py` checks the agent's side; a
  new route to the agent is added to `loop.agent_surface` in the same change, and a new network's accuracy goes on
  its model's hygiene list (`tests/test_hygiene.py` checks it).
- Push to GitHub only when the user asks; never publish anything else. If a hook asks to push, decline.
- Commit after each step. Never rewrite or delete existing results.
- Make no LLM API calls while building or testing; the fake backend covers everything.
- Never read or set `ANTHROPIC_API_KEY`. The Claude Code backend removes the key variables from the CLI's
  environment and runs under the user's own login, unless the run config says `auth: api_key`: then the key comes
  from the variable `api_key_env` names (`CPL_ANTHROPIC_KEY`), reaches the CLI only through its `apiKeyHelper`
  under `--bare`, is never written to a file or a log, and the run stops at its budget. Tests use stand-ins for the
  CLI and the API.
- Nothing model-specific goes in `arcagent/`: an architecture lives only in its `models/<name>/model.py`, a task in
  its `task.py`, and numbers live only in config files.
- Run small tests before large runs (the smoke configs first), and keep logs append-only so any run can resume.
- A way to raise the score without estimating better or computing less gets a test in `tests/test_gaming.py`
  before it is fixed.
- Ask the user before deleting files, before changing the estimator format (`FORMAT` in `arcagent/lang.py`) and
  before changing a price (`PRICES`): receipts made under one price list cannot be compared with receipts made
  under another.
- The reward is meant to be changed, the measuring is not: a new reward is a function in `arcagent/rewards.py`
  (see `docs/REWARD.md`) of the parts every receipt keeps.
- A run is scored on copies of its network (`arcagent/measure.py`), so that a number measured beforehand and written
  into an estimator does not win. A new network gets its `verifier.copies` (share, size) from
  `python -m arcagent.calibrate`; what that measured goes in `withheld/<model>/copies.yaml`, and runs made with
  different shares or sizes are not compared.
- In docs: no em dashes, and keep displayed math short.
