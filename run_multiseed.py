"""Multi-seed reliability of the Config B deep-RL agents.

Each agent is trained from several random seeds (with the same SOFA reward shaping
and Optuna hyperparameters as the main run) and evaluated on a fixed set of
evaluation seeds. Reporting the mean and standard deviation across training seeds
separates the agent's reliable performance from single-run luck, as recommended
for deep RL by Henderson et al. (2018). Only the training seed varies; the
evaluation seeds are held fixed, so every run is scored on the same patients.

Outputs (for the report; the notebooks are not touched):
  configB_multiseed_results.json - per-seed and aggregated metrics
  plots/configB_multiseed.png    - survival per agent with across-seed error bars
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sepsis_rl as srl

# Configuration. 300k steps keeps the 6 runs tractable (~2h); the across-seed
# spread, not the peak performance, is what this experiment measures. Add A2C to
# ALGOS for a third agent (three more runs).
SEEDS = [42, 7, 123]
ALGOS = ["DQN", "PPO"]          # add "A2C" for the third agent
TIMESTEPS = 300_000
EVAL_FREQ = 50_000
N_EVAL_TRAIN = 50
N_EVAL_FINAL = 1000
BUCKETS_REPORTED = ["All", "Clean", "Noisy", "Missing"]

results_path = Path("configB_multiseed_results.json")
summary = {"metadata": {"seeds": SEEDS, "algos": ALGOS, "timesteps": TIMESTEPS,
                        "shaping": True, "use_tuned": True,
                        "started_at": datetime.now().isoformat(timespec="seconds")},
           "per_seed": {}, "aggregated": {}}

# Fixed reference policies (deterministic given the fixed eval seeds).
print("=== reference baselines ===", flush=True)
summary["random"] = srl.random_baseline_by_condition(n_episodes=N_EVAL_FINAL)
summary["expert"] = srl.expert_baseline_by_condition(n_episodes=N_EVAL_FINAL)

for algo in ALGOS:
    tuned = srl.load_tuned_hp(algo)
    summary["per_seed"][algo] = {}
    for seed in SEEDS:
        tag = f"{algo.lower()}_s{seed}"
        print(f"\n=== TRAIN {algo} seed={seed} ({TIMESTEPS:,} steps) "
              f"@ {datetime.now().isoformat(timespec='seconds')} ===", flush=True)
        srl.train_agent(algo, timesteps=TIMESTEPS, normalize=True, seed=seed, tag=tag,
                        eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_TRAIN,
                        progress_bar=False, verbose=0, shaping=True, shaping_beta=0.05,
                        hyperparams=tuned)
        res = srl.evaluate_conditions(tag, algo, n_episodes=N_EVAL_FINAL)
        summary["per_seed"][algo][str(seed)] = res
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Aggregate across seeds.
    agg = {}
    for b in BUCKETS_REPORTED:
        survs = [summary["per_seed"][algo][str(s)][b]["survival"]
                 for s in SEEDS if b in summary["per_seed"][algo][str(s)]]
        rets = [summary["per_seed"][algo][str(s)][b]["return"]
                for s in SEEDS if b in summary["per_seed"][algo][str(s)]]
        ints = [summary["per_seed"][algo][str(s)][b]["intensity"]
                for s in SEEDS if b in summary["per_seed"][algo][str(s)]]
        if survs:
            agg[b] = {"survival_mean": float(np.mean(survs)),
                      "survival_std": float(np.std(survs)),
                      "return_mean": float(np.mean(rets)),
                      "return_std": float(np.std(rets)),
                      "intensity_mean": float(np.mean(ints))}
    summary["aggregated"][algo] = agg
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[{algo}] across {len(SEEDS)} seeds: "
          f"All survival = {agg['All']['survival_mean']*100:.1f}% "
          f"+/- {agg['All']['survival_std']*100:.1f}", flush=True)

# Figure: All vs Clean survival per agent, error bars = std across seeds.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
rand_all = summary["random"]["All"]["survival"] * 100
rand_clean = summary["random"]["Clean"]["survival"] * 100
exp_all = summary["expert"]["All"]["survival"] * 100
exp_clean = summary["expert"]["Clean"]["survival"] * 100
for ax, bucket, rand_ref, exp_ref in [(axes[0], "All", rand_all, exp_all),
                                      (axes[1], "Clean", rand_clean, exp_clean)]:
    means = [summary["aggregated"][a][bucket]["survival_mean"] * 100 for a in ALGOS]
    stds = [summary["aggregated"][a][bucket]["survival_std"] * 100 for a in ALGOS]
    x = np.arange(len(ALGOS))
    ax.bar(x, means, yerr=stds, capsize=5, color="#2980b9", alpha=0.85)
    ax.axhline(rand_ref, ls="--", color="gray", label=f"Random ({rand_ref:.1f}%)")
    ax.axhline(exp_ref, ls=":", color="#e67e22", label=f"Expert ({exp_ref:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(ALGOS)
    ax.set_title(f"{bucket} episodes")
    ax.set_ylabel("Survival rate (%)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.suptitle(f"Config B - across-seed reliability ({len(SEEDS)} seeds, "
             f"{TIMESTEPS//1000}k steps)", fontsize=12)
fig.tight_layout()
fig.savefig("plots/configB_multiseed.png", dpi=150, bbox_inches="tight")
print("Saved plots/configB_multiseed.png")

summary["metadata"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nDone. See configB_multiseed_results.json and plots/configB_multiseed.png.")
