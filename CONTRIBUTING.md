# Contributing

This is a collaborative academic artifact. Contributions should preserve experimental provenance and avoid rewriting reported results without regenerated evidence.

1. Create a branch from `main`.
2. Add tests for changes to the environments, wrappers, or evaluation protocol.
3. Run `python -m unittest discover -s tests -v`.
4. Regenerate the affected result files and figures with the `run_*.py` scripts, and state the seeds and training budget used.
5. Open a pull request and credit all contributors to new results.

Evaluation must always run on the true reward and the default `make_clinical_env()` parameters. Training-only levers such as reward shaping or tuned hyperparameters must be documented as such.

Do not commit student numbers, private identifiers, patient-level data, or generated model weights.
