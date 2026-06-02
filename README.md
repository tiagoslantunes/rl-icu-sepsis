# Sepsis Treatment Optimization via Reinforcement Learning

Reinforcement Learning course project (Master in Data Science & Advanced Analytics).
We learn ICU sepsis treatment policies (vasopressor + IV-fluid dosing) on the
`Sepsis/ICU-Sepsis-v2` benchmark (MIMIC-III derived), balancing **patient
survival** against **treatment intensity** (a parsimony penalty `lam=0.02`).

## Project structure

```
envs/
  env_setup.py             # constants + make_sepsis_env() (Config A, discrete MDP)
  continuous_sepsis_env.py # ContinuousICUSepsisEnv (Config B, 47-dim observations)
  wrappers.py              # clinical failure-mode wrappers + make_clinical_env()
  tabular_agents.py        # Config A agents: PolicyIteration, QLearning, SARSA
sepsis_rl.py               # Config B: DQN/PPO/A2C training, evaluation, plots, Optuna
rl_sepsis_project.ipynb    # Config A notebook (tabular methods + analysis)
rl_sepsis_projectB.ipynb   # Config B notebook (calls sepsis_rl.py)
configA_results.json       # real Config A metrics, produced by the Config A notebook
requirements.txt
plots/                     # all figures are written here
```

## Configurations

- **Configuration A — Tabular RL.** Discrete MDP, 716 states × 25 actions. The full
  model (`P`, `R`) is available, so we run **Policy Iteration** (model-based),
  **Q-Learning** and **SARSA** (model-free TD). The Q-table is only 17,900 entries.
- **Configuration B — Continuous Deep RL.** 47-dim continuous physiological
  observations with three clinical failure-mode wrappers: episodic observation
  noise, episodic missing labs, and rare acute deterioration events. We train
  **DQN**, **PPO** and **A2C** (with/without observation normalization).

## Reproducibility / how to run

1. `pip install -r requirements.txt`
2. Run the notebooks **in Jupyter / UTF-8** (the env prints contain Unicode; a raw
   Windows `cp1252` console can raise `UnicodeEncodeError`).
3. **Config A first** (`rl_sepsis_project.ipynb`): runs end-to-end in ~minutes and
   writes the real metrics to `configA_results.json`.
4. **Config B** (`rl_sepsis_projectB.ipynb`): `TIMESTEPS=150_000` is a fast preview;
   set `1_000_000` for report-quality runs (several hours; GPU recommended).
   The Config B comparison table reads `configA_results.json`, so run Config A first.

All reported numbers use **fixed evaluation seeds**, deterministic policies, and the
**default** `make_clinical_env()` parameters. The clinical wrappers are never altered
for main results; robustness is measured by bucketing episodes via the `info` flags.

The branch includes the final generated report figures under `plots/` so the
analysis can be inspected without rerunning the deep-RL training.

### Config B 1M run

For report-quality deep-RL evidence, `run_configB_1m.py` trains the three
strongest normalized Config B candidates for `1_000_000` timesteps each:
`DQN-v2 1M`, `PPO-v2 1M`, and `A2C 1M`. The generated outputs are committed as:

- `configB_1m_results.json`
- `configB_1m_compare_configs.csv`
- `plots/configB_1m_*.png`

In the 1M run, `PPO-v2 1M` is the best Config B policy by survival/return
(`0.5955` mean return, `67.8%` survival), but it still remains below the
tabular Config A methods. This is an important result: the continuous, noisy,
partially missing clinical setting is substantially harder than the discrete
known-MDP setting.

## Creative extension

**Robustness + clinical interpretability audit:** per-failure-mode degradation
(Clean / Noisy / Missing / Acute), DQN feature-importance via Q-value perturbation,
and treatment-intensity / dose-grid analysis of the learned policies.

> Disclaimer: results are on a **simulated** benchmark and are **not** clinical advice.
