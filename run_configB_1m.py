
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sepsis_rl as srl

TIMESTEPS = 1_000_000
EVAL_FREQ = 50_000
N_EVAL_TRAIN = 100
N_EVAL_FINAL = 1000
RUNS = [
    ("DQN", "dqn_v2_1m", "DQN-v2 1M"),
    ("PPO", "ppo_v2_1m", "PPO-v2 1M"),
    ("A2C", "a2c_v2_1m", "A2C 1M"),
]

Path("plots").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
summary = {"metadata": {
    "timesteps": TIMESTEPS,
    "eval_freq": EVAL_FREQ,
    "n_eval_train": N_EVAL_TRAIN,
    "n_eval_final": N_EVAL_FINAL,
    "started_at": datetime.now().isoformat(timespec="seconds"),
}}
results = {}
tags_labels = {}

random_stats = srl.random_baseline(n_episodes=N_EVAL_FINAL)
summary["random"] = random_stats

for algo, tag, label in RUNS:
    print(f"\n=== TRAIN {label} ({TIMESTEPS:,} steps) @ {datetime.now().isoformat(timespec='seconds')} ===", flush=True)
    srl.train_agent(
        algo,
        timesteps=TIMESTEPS,
        normalize=True,
        tag=tag,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_TRAIN,
        progress_bar=False,
        verbose=0,
    )
    print(f"=== EVALUATE {label} ===", flush=True)
    res = srl.evaluate_conditions(tag, algo, n_episodes=N_EVAL_FINAL)
    results[label] = res
    tags_labels[tag] = label
    summary[label] = res
    Path("configB_1m_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n=== GENERATE 1M FIGURES ===", flush=True)
srl.plot_learning_curves(tags_labels, baseline=random_stats["return"],
                         filename="configB_1m_learning_curves.png", show=False)
srl.plot_learning_curves(tags_labels, baseline=random_stats["return"],
                         zoom_steps=300_000,
                         filename="configB_1m_learning_curves_zoom.png", show=False)
srl.plot_robustness(results, metric="return", baseline=random_stats["return"],
                    filename="configB_1m_robustness_return.png", show=False)
srl.plot_robustness(results, metric="survival", baseline=random_stats["survival"],
                    filename="configB_1m_robustness_survival.png", show=False)
srl.plot_dose_grid(results, filename="configB_1m_dose_grid.png", show=False)
try:
    srl.feature_importance_dqn("dqn_v2_1m", filename="configB_1m_feature_importance.png", show=False)
except Exception as exc:
    print(f"Feature importance skipped/failed: {exc}", flush=True)

# Treatment intensity bar chart for final 1M policies.
import numpy as np
labels = list(results.keys())
vals = [results[label]["All"]["intensity"] for label in labels]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, vals)
ax.set_ylabel("Mean treatment intensity / step")
ax.set_title("Config B 1M - Treatment intensity")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/configB_1m_intensity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Honest Config A vs Config B 1M table.
comparison = srl.compare_configs(results)
print("\n=== CONFIG A VS CONFIG B 1M ===")
print(comparison.to_string(index=False))
comparison.to_csv("configB_1m_compare_configs.csv", index=False)

summary["metadata"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
Path("configB_1m_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

for p in Path("outputs").glob("configB_1m_*.png"):
    shutil.copy2(p, Path("plots") / p.name)
print("\nSaved configB_1m_results.json, configB_1m_compare_configs.csv, and plots/configB_1m_*.png", flush=True)
