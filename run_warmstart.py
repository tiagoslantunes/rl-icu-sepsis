"""Learning from demonstrations (Config A): expert Q warm-start.

Compares tabular Q-learning trained from zeros against the same agent initialised
from expert demonstrations (Monte Carlo returns of the clinician policy, the
demonstration pre-training stage of DQfD, Hester et al. 2018). Produces the
sample-efficiency curve (greedy survival vs training episodes) and the number of
episodes each variant needs to reach a target, with Policy Iteration as the
model-based ceiling.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.env_setup import GAMMA, make_sepsis_env
from envs.tabular_agents import (QLearning, PolicyIteration, evaluate_policy,
                                 expert_warmstart_q)

SEED = 42
N_EPISODES = 15_000
EVAL_EVERY = 1_000
EVAL_EPS = 300

Path("plots").mkdir(exist_ok=True)

# Model-based ceiling for reference.
env = make_sepsis_env(); raw = env.unwrapped
P = raw._tx_mat; R = (P * raw._r_mat).sum(axis=2); env.close()
pi = PolicyIteration(P, R, gamma=GAMMA); pi.train(max_iterations=200, verbose=False)
pi_surv = evaluate_policy(pi.get_policy(), n_episodes=1000, seed=SEED)["survival_rate"] * 100

# Expert demonstration warm-start.
Q_init = expert_warmstart_q(n_episodes=3000, gamma=GAMMA, seed=SEED)
coverage = float((Q_init != 0).any(axis=1).mean())
print(f"Warm-start seeded action values for {coverage*100:.0f}% of states.")

# Cold start (zeros) vs warm start (expert demonstrations).
cold = QLearning(seed=SEED, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_steps=N_EPISODES)
cold_h = cold.train(n_episodes=N_EPISODES, eval_every=EVAL_EVERY,
                    eval_eps=EVAL_EPS, verbose=False)

warm = QLearning(seed=SEED, epsilon_start=0.5, epsilon_end=0.01,
                 epsilon_decay_steps=N_EPISODES)
warm.Q = Q_init.copy()
warm_h = warm.train(n_episodes=N_EPISODES, eval_every=EVAL_EVERY,
                    eval_eps=EVAL_EPS, verbose=False)

# Sample efficiency: episodes to reach a fixed clinically meaningful survival
# target (70%, between the random baseline ~68% and the Policy Iteration ceiling).
TARGET = 70.0
def episodes_to(hist, target):
    surv = hist["eval_survival"] * 100
    hit = np.where(surv >= target)[0]
    return int(hist["eval_steps"][hit[0]]) if len(hit) else None
cold_ep = episodes_to(cold_h, TARGET)
warm_ep = episodes_to(warm_h, TARGET)

# Early-training advantage at a fixed checkpoint.
def survival_at(hist, episode):
    i = int(np.argmin(np.abs(hist["eval_steps"] - episode)))
    return float(hist["eval_survival"][i] * 100)
early_cold = survival_at(cold_h, 2000)
early_warm = survival_at(warm_h, 2000)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cold_h["eval_steps"], cold_h["eval_survival"] * 100,
        marker="o", ms=3, label="Q-learning (cold start)")
ax.plot(warm_h["eval_steps"], warm_h["eval_survival"] * 100,
        marker="s", ms=3, label="Q-learning (expert warm-start)")
ax.axhline(pi_surv, ls="--", lw=1.2, color="green",
           label=f"Policy Iteration ceiling ({pi_surv:.1f}%)")
ax.axhline(TARGET, ls=":", lw=1.0, color="gray", label=f"Target ({TARGET:.0f}%)")
ax.set_xlabel("Training episodes")
ax.set_ylabel("Greedy survival rate (%)")
ax.set_title("Config A - Learning from demonstrations: warm-start sample efficiency")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/configA_warmstart.png", dpi=150, bbox_inches="tight")

cold_final = evaluate_policy(cold.get_policy(), n_episodes=1000, seed=SEED)
warm_final = evaluate_policy(warm.get_policy(), n_episodes=1000, seed=SEED)
summary = {
    "policy_iteration_survival": pi_surv,
    "warmstart_state_coverage": coverage,
    "target_survival": TARGET,
    "cold": {"episodes_to_target": cold_ep,
             "survival_at_2000ep": early_cold,
             "final_survival": cold_final["survival_rate"],
             "final_intensity": cold_final["mean_intensity"]},
    "warm": {"episodes_to_target": warm_ep,
             "survival_at_2000ep": early_warm,
             "final_survival": warm_final["survival_rate"],
             "final_intensity": warm_final["mean_intensity"]},
}
Path("configA_warmstart_results.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"\nEarly advantage (at 2000 episodes): warm {early_warm:.1f}% vs cold "
      f"{early_cold:.1f}% survival.")
print(f"Episodes to reach {TARGET:.0f}% survival: "
      f"warm={warm_ep}, cold={cold_ep}.")
print("Saved plots/configA_warmstart.png and configA_warmstart_results.json.")