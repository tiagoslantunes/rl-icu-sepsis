<div align="center">

# Sepsis Treatment Optimization via Reinforcement Learning

**Learning ICU vasopressor and IV-fluid dosing policies that trade survival against treatment intensity.**
Tabular RL &middot; deep RL &middot; clinical failure-mode robustness &middot; potential-based reward shaping.

[![Quality checks](https://github.com/tiagoslantunes/rl-icu-sepsis/actions/workflows/quality.yml/badge.svg)](https://github.com/tiagoslantunes/rl-icu-sepsis/actions/workflows/quality.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN_|_PPO_|_A2C-EE4C2C?logo=pytorch&logoColor=white)](https://stable-baselines3.readthedocs.io/)
[![Benchmark](https://img.shields.io/badge/benchmark-ICU--Sepsis--v2-00A36C)](https://github.com/icu-sepsis/icu-sepsis)
[![License](https://img.shields.io/badge/license-all_rights_reserved-6c757d)](LICENSE)
[![Upstream](https://img.shields.io/badge/fork-upstream-6f42c1?logo=github)](https://github.com/mariatavarespimentel/RL-ICU-Sepsis)

</div>

> [!IMPORTANT]
> This is Tiago Antunes's portfolio fork of a collaborative Reinforcement Learning course
> project. Full team attribution is preserved below and upstream provenance stays visible in
> GitHub's fork metadata.

Reinforcement Learning course project for the MSc in Data Science & Advanced Analytics at
NOVA IMS. We learn ICU sepsis treatment policies on the `Sepsis/ICU-Sepsis-v2` benchmark
(MIMIC-III derived), balancing **patient survival** against **treatment intensity** through a
parsimony penalty `lam=0.02`.

> [!WARNING]
> Results are produced on a **simulated** benchmark and are **not** clinical advice. No
> patient-level data is contained in or redistributed by this repository.

## Highlights

- Two configurations on the same problem: a fully known 716-state discrete MDP and a 47-dimensional continuous, noisy, partially-observed version.
- Policy Iteration gives the exact tabular optimum, so every other agent is measured against a known ceiling.
- Three clinical failure-mode wrappers — observation noise, missing labs, acute deterioration — with per-mode degradation reported separately.
- Return is decomposed into survival vs treatment intensity, which exposes agents that "win" only by treating less.
- SOFA potential-based reward shaping (Ng et al., 1999) redirects the policy toward survival with a dense, policy-invariant signal.
- Bootstrap 95% confidence intervals and multi-seed reliability on every headline number.

## Project structure

| Path | Purpose |
|---|---|
| [`envs/`](envs) | Environment setup, continuous env, clinical wrappers, and tabular agents |
| [`sepsis_rl.py`](sepsis_rl.py) | Config B: DQN/PPO/A2C training, evaluation, plots, and Optuna tuning |
| [`rl_sepsis_configA.ipynb`](rl_sepsis_configA.ipynb) | Config A notebook — tabular methods and analysis |
| [`rl_sepsis_configB.ipynb`](rl_sepsis_configB.ipynb) | Config B notebook — calls `sepsis_rl.py` |
| [`run_multiseed.py`](run_multiseed.py) | Multi-seed reliability of the Config B agents |
| [`run_bc.py`](run_bc.py) | Behavioural cloning of the clinician policy |
| [`run_upgrades.py`](run_upgrades.py) | Pareto plot, Config A bootstrap CIs, Q-value overestimation diagnostic |
| [`add_expert_ci.py`](add_expert_ci.py) | Post-hoc re-evaluation with bootstrap confidence intervals |
| [`regenerate_configB.py`](regenerate_configB.py) | Rebuild headline results and figures without retraining |
| [`plots/`](plots) | Committed figures for every reported result |
| [`report/`](report) | LaTeX report, bundled figures, and the compiled PDF |
| [`archive/`](archive) | Superseded notebook kept for provenance |
| [`tests/`](tests) | Dependency-free consistency checks over the committed result files |

Result files (`configA_results.json`, `configB_1m_results.json`, `configB_multiseed_results.json`,
`configB_overestimation.json`, `configB_1m_compare_configs.csv`) stay at the repository root
because the scripts and notebooks read them from there.

## Configurations

- **Configuration A — Tabular RL.** Discrete MDP, 716 states × 25 actions. The full model
  (`P`, `R`) is available, so we run **Policy Iteration** (model-based), **Q-Learning** and
  **SARSA** (model-free TD). The Q-table is only 17,900 entries.
- **Configuration B — Continuous Deep RL.** 47-dimensional continuous physiological
  observations with three clinical failure-mode wrappers: episodic observation noise, episodic
  missing labs, and rare acute deterioration events. We train **DQN**, **PPO** and **A2C**,
  with and without observation normalization.

## Quick start

```bash
git clone https://github.com/tiagoslantunes/rl-icu-sepsis.git
cd rl-icu-sepsis
python -m pip install -r requirements.txt
```

Run the notebooks in Jupyter with a UTF-8 console — the environment prints contain Unicode, and
a raw Windows `cp1252` console raises `UnicodeEncodeError`.

1. **Config A first** ([`rl_sepsis_configA.ipynb`](rl_sepsis_configA.ipynb)) — runs end-to-end
   in minutes and writes the real metrics to `configA_results.json`.
2. **Config B** ([`rl_sepsis_configB.ipynb`](rl_sepsis_configB.ipynb)) — `TIMESTEPS=150_000`
   is a fast preview; `1_000_000` gives report-quality runs (several hours, GPU recommended).
   The Config B comparison table reads `configA_results.json`, so run Config A first.

The committed figures under `plots/` mean the analysis can be inspected without rerunning any
deep-RL training.

## Reproducibility

- All reported numbers use fixed evaluation seeds, deterministic policies, and the default
  `make_clinical_env()` parameters.
- The clinical wrappers are never altered for main results; robustness is measured by bucketing
  episodes through the `info` flags.
- The 1M run uses two **training-only** levers — SOFA reward shaping (`SHAPING=True`) and the
  Optuna best hyperparameters (`USE_TUNED=True`). Evaluation always stays on the true reward
  and the default environment.
- `regenerate_configB.py` rebuilds the headline results and figures from the committed tuned
  models without retraining.

## Results

ICU-Sepsis has a deliberately **high** random baseline and a **small** optimal-vs-random gap:
Config A Policy Iteration, the exact optimum, reaches only ~79% survival against ~69% for
random. Against a per-condition random baseline, the unshaped deep agents barely improve
*survival* — they mainly learn lower *treatment intensity*, earning return through the `lam`
penalty rather than through more survivors.

We therefore report return decomposed into survival and intensity, and do not overclaim. The
continuous, noisy, partially-missing clinical setting is genuinely harder than the discrete
known-MDP setting.

## Creative extension

Robustness, clinical interpretability, and reward design:

- Per-failure-mode degradation (Clean / Noisy / Missing / Acute) against a fair, bucketed random baseline.
- DQN feature importance via Q-value perturbation over sepsis-severity markers.
- Treatment-intensity and 5×5 dose-grid analysis of the learned policies.
- SOFA potential-based reward shaping (`SofaShapingEnv`, Ng et al., 1999), which diagnoses that naive RL optimises parsimony over survival and redirects the policy toward survival.

## Limitations

- The benchmark is a simulator: no result here transfers to bedside decisions without prospective clinical validation.
- The optimal-vs-random gap is small by construction, so deep-RL gains are correspondingly modest and easy to overstate.
- Deep agents are evaluated on fixed seeds; multi-seed spread is reported but the seed budget is limited by training cost.
- Behavioural cloning imitates the logged clinician policy and inherits whatever bias that policy carries.

## Quality checks

Every push runs [`quality.yml`](.github/workflows/quality.yml) on GitHub Actions: a syntax
check, notebook JSON validation, and dependency-free consistency tests over the committed
result files — no RL dependencies and no retraining required. To run the same checks locally:

```bash
python -m compileall -q envs sepsis_rl.py tests
python -m unittest discover -s tests -v
```

## Authors

- Maria Pimentel
- Gonçalo Arrobas
- Tiago Antunes

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

No open-source license has been granted for the original code, report, or figures; see
[LICENSE](LICENSE). Reuse beyond what copyright law permits requires the authors' prior written
permission. The ICU-Sepsis benchmark and all third-party libraries remain subject to their own
terms.
