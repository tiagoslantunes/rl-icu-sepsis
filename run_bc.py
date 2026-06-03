"""Learning from demonstrations: behavioural cloning of the clinician policy.

Trains a neural policy to imitate the AI Clinician behaviour policy from the
47-dimensional continuous observations (the imitation / pre-training stage of
demonstration-based deep RL such as DQfD, Hester et al. 2018), evaluates it with
the same bucketed, CI-reported protocol as the agents, and adds it to the Config B
comparison. The trained agents are reused from the saved 1M results, so this does
not retrain any RL agent.
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import sepsis_rl as srl

# 1. Behavioural cloning from expert demonstrations
X, y = srl.collect_expert_demonstrations(n_episodes=4000)
print(f"Collected {len(X)} demonstration steps from the expert policy.")
bc = srl.train_bc(X, y, epochs=30)
bc_res = srl.evaluate_bc(bc, n_episodes=1000)

# 2. Record the BC result alongside the existing 1M evaluation
results_path = Path("configB_1m_results.json")
summary = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
summary["bc"] = bc_res
results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# 3. Assemble the full comparison (reference policies + imitation + RL agents)
order = [("random", "Random"), ("expert", "Expert"), ("bc", "BC (imitation)"),
         ("DQN-v2 1M", "DQN-v2 1M"), ("PPO-v2 1M", "PPO-v2 1M"), ("A2C 1M", "A2C 1M")]
results = {label: summary[key] for key, label in order if key in summary}

comparison = srl.compare_configs(results)
print("\n=== CONFIG A vs CONFIG B (with BC and 95% CI on Config B survival) ===")
print(comparison.to_string(index=False))
comparison.to_csv("configB_1m_compare_configs.csv", index=False)

srl.plot_robustness(results, metric="survival",
                    baseline=results["Random"]["All"]["survival"],
                    filename="configB_1m_robustness_survival.png", show=False)
srl.plot_dose_grid(results, filename="configB_1m_dose_grid.png", show=False)

for p in Path("outputs").glob("configB_1m_*.png"):
    shutil.copy2(p, Path("plots") / p.name)
print("\nUpdated configB_1m_results.json, configB_1m_compare_configs.csv and plots/.")
