"""Post-hoc evaluation of the trained 1M models.

Re-evaluates the existing Config B agents (no retraining) together with the
random and clinician-expert reference policies, attaching bootstrap 95%
confidence intervals to every reported metric. Regenerates the results file, the
Config A vs Config B comparison table and the comparison figures (including the
expert reference line).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import sepsis_rl as srl

N_EVAL = 1000
AGENTS = [("dqn_v2_1m", "DQN", "DQN-v2 1M"),
          ("ppo_v2_1m", "PPO", "PPO-v2 1M"),
          ("a2c_v2_1m", "A2C", "A2C 1M")]

results = {}
results["Random"] = srl.random_baseline_by_condition(n_episodes=N_EVAL)
results["Expert"] = srl.expert_baseline_by_condition(n_episodes=N_EVAL)
for tag, algo, label in AGENTS:
    results[label] = srl.evaluate_conditions(tag, algo, n_episodes=N_EVAL)

# Preserve the training metadata produced by run_configB_1m.py, if present.
out_path = Path("configB_1m_results.json")
summary = {}
if out_path.exists():
    summary = json.loads(out_path.read_text(encoding="utf-8"))
summary["random"] = results["Random"]
summary["expert"] = results["Expert"]
for tag, algo, label in AGENTS:
    summary[label] = results[label]
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

comparison = srl.compare_configs(results)
print("\n=== CONFIG A vs CONFIG B (with 95% CI on Config B survival) ===")
print(comparison.to_string(index=False))
comparison.to_csv("configB_1m_compare_configs.csv", index=False)

srl.plot_robustness(results, metric="survival",
                    baseline=results["Random"]["All"]["survival"],
                    filename="configB_1m_robustness_survival.png", show=False)
srl.plot_robustness(results, metric="return",
                    baseline=results["Random"]["All"]["return"],
                    filename="configB_1m_robustness_return.png", show=False)
srl.plot_robustness(results, metric="intensity",
                    filename="configB_1m_intensity.png", show=False)
srl.plot_dose_grid(results, filename="configB_1m_dose_grid.png", show=False)

import shutil
for p in Path("outputs").glob("configB_1m_*.png"):
    shutil.copy2(p, Path("plots") / p.name)
print("\nUpdated configB_1m_results.json, configB_1m_compare_configs.csv and plots/.")
