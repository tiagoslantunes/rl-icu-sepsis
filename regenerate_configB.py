"""Regenerate the Config B headline results and figures from the committed 1M
tuned models, without retraining.

The three deep agents (dqn_tuned, ppo_tuned, a2c_tuned) are the full-budget
(1M-step) Optuna-selected runs. They are evaluated with the standard protocol
(bucketed by failure mode, fixed evaluation seeds, bootstrap 95% CIs) and all
report figures are regenerated from their logs and policies. The behavioural
cloning result is reused from the previous run since it is independent of the
deep agents.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sepsis_rl as srl

N_EVAL = 1000
# (algorithm, on-disk tag, report label/key)
RUNS = [("DQN", "dqn_tuned", "DQN-v2 1M"),
        ("PPO", "ppo_tuned", "PPO-v2 1M"),
        ("A2C", "a2c_tuned", "A2C 1M")]

Path("plots").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# Reference policies.
print("=== reference baselines ===", flush=True)
random_buckets = srl.random_baseline_by_condition(n_episodes=N_EVAL)
expert_buckets = srl.expert_baseline_by_condition(n_episodes=N_EVAL)

# Reuse the BC result if present, else recompute.
prev = {}
if Path("configB_1m_results.json").exists():
    prev = json.loads(Path("configB_1m_results.json").read_text(encoding="utf-8"))
bc_buckets = prev.get("bc")
if bc_buckets is None:
    print("=== behavioural cloning (recompute) ===", flush=True)
    X, y = srl.collect_expert_demonstrations(n_episodes=4000)
    bc = srl.train_bc(X, y, epochs=30)
    bc_buckets = srl.evaluate_bc(bc, n_episodes=N_EVAL)

# Evaluate the tuned 1M agents.
results, tags_labels, summary = {}, {}, {"metadata": prev.get("metadata", {})}
summary["metadata"]["source"] = "1M Optuna-tuned models (dqn/ppo/a2c_tuned)"
results["Random"] = random_buckets
results["Expert"] = expert_buckets
summary["random"] = random_buckets
summary["expert"] = expert_buckets
summary["bc"] = bc_buckets
for algo, tag, label in RUNS:
    print(f"=== EVALUATE {label} ({tag}) ===", flush=True)
    res = srl.evaluate_conditions(tag, algo, n_episodes=N_EVAL)
    results[label] = res
    summary[label] = res
    tags_labels[tag] = label.replace(" 1M", "")
Path("configB_1m_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Figures (from the tuned logs / policies) 
print("=== figures ===", flush=True)
srl.plot_learning_curves(tags_labels, baseline=random_buckets["All"]["return"],
                         show_trend=True, filename="configB_1m_learning_curves.png",
                         show=False)
srl.plot_learning_curves(tags_labels, baseline=random_buckets["All"]["return"],
                         zoom_steps=300_000, show_trend=True,
                         filename="configB_1m_learning_curves_zoom.png", show=False)
srl.plot_curves_grid(tags_labels, baseline=random_buckets["All"]["return"],
                     filename="configB_tuned_grid.png", show=False)

# Comparison plots include random + expert + bc + the three agents.
plot_results = {"Random": random_buckets, "Expert": expert_buckets,
                "BC (imitation)": bc_buckets,
                "DQN": results["DQN-v2 1M"], "PPO": results["PPO-v2 1M"],
                "A2C": results["A2C 1M"]}
srl.plot_robustness(plot_results, metric="survival",
                    baseline=random_buckets["All"]["survival"],
                    filename="configB_1m_robustness_survival.png", show=False)
srl.plot_robustness(plot_results, metric="return",
                    baseline=random_buckets["All"]["return"],
                    filename="configB_1m_robustness_return.png", show=False)
srl.plot_dose_grid({"DQN": results["DQN-v2 1M"], "PPO": results["PPO-v2 1M"],
                    "A2C": results["A2C 1M"], "Expert": expert_buckets},
                   filename="configB_1m_dose_grid.png", show=False)
srl.feature_importance_dqn("dqn_tuned", filename="configB_1m_feature_importance.png",
                           show=False)
try:
    srl.shap_importance_dqn("dqn_tuned", filename="configB_shap_importance.png",
                            show=False)
except Exception as exc:
    print(f"SHAP skipped: {exc}", flush=True)

# Treatment-intensity bar chart.
labels = ["Random", "Expert", "BC (imitation)", "DQN", "PPO", "A2C"]
vals = [plot_results[l]["All"]["intensity"] for l in labels]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, vals, color="#2980b9")
ax.set_ylabel("Mean treatment intensity / step")
ax.set_title("Config B (1M) - treatment intensity")
ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig("outputs/configB_1m_intensity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Honest comparison table.
comparison = srl.compare_configs(plot_results)
print("\n=== CONFIG A vs CONFIG B (tuned 1M) ===")
print(comparison.to_string(index=False))
comparison.to_csv("configB_1m_compare_configs.csv", index=False)

import shutil
for p in Path("outputs").glob("configB_1m_*.png"):
    shutil.copy2(p, Path("plots") / p.name)
print("\nDone. Updated configB_1m_results.json, CSV and plots/.")