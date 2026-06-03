"""Three analysis upgrades that reuse the already-trained models and results.

1. Treatment-intensity vs survival Pareto plot (all policies, both configs).
2. Config A bootstrap 95% CIs (re-trains the fast tabular agents only).
3. DQN Q-value overestimation diagnostic (loads the saved 1M model; no retraining).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("plots").mkdir(exist_ok=True)
SEED = 42


# --------------------------------------------------------------------------- #
# 2. Config A bootstrap CIs  (also needed for the Pareto error bars)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return float("nan")
    rng = np.random.RandomState(seed)
    boot = a[rng.randint(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float((hi - lo) / 2.0)


def config_a_with_ci():
    from envs.env_setup import GAMMA, make_sepsis_env, N_STATES, N_ACTIONS
    from envs.tabular_agents import (PolicyIteration, QLearning, SARSA,
                                     evaluate_policy)
    env = make_sepsis_env(); raw = env.unwrapped
    P = raw._tx_mat; R = (P * raw._r_mat).sum(axis=2); env.close()

    # Reproduce the notebook training exactly (same optimistic init, episodes,
    # epsilon schedule AND periodic-evaluation cadence, so the RNG path matches).
    pi = PolicyIteration(P, R, gamma=GAMMA); pi.train(max_iterations=200, verbose=False)
    np.random.seed(SEED)
    ql = QLearning(seed=SEED, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=35_000)
    ql.Q[:] = 1.0; ql.train(n_episodes=50_000, eval_every=5_000, eval_eps=300, verbose=False)
    np.random.seed(SEED)
    sa = SARSA(seed=SEED, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=35_000)
    sa.Q[:] = 1.0; sa.train(n_episodes=50_000, eval_every=5_000, eval_eps=300, verbose=False)
    np.random.seed(SEED)
    rand_policy = np.random.randint(0, N_ACTIONS, size=N_STATES)

    out = {}
    for name, pol in [("random", rand_policy), ("policy_iteration", pi.get_policy()),
                      ("q_learning", ql.get_policy()), ("sarsa", sa.get_policy())]:
        r = evaluate_policy(pol, n_episodes=1000, seed=SEED)
        survived = r["survivals"]   # per-episode survival (terminal-reward based)
        out[name] = {
            "mean_return": r["mean_return"],
            "survival_rate": r["survival_rate"],
            "survival_ci": _bootstrap_ci(survived),
            "mean_ep_length": r["mean_ep_length"],
            "mean_intensity": r["mean_intensity"],
        }
        print(f"[A] {name:18s} surv={r['survival_rate']*100:5.1f}% "
              f"+/-{out[name]['survival_ci']*100:.1f}  intens={r['mean_intensity']:.3f}")
    Path("configA_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# 1. Pareto: treatment intensity vs survival
# --------------------------------------------------------------------------- #
def pareto_plot(config_a):
    config_b = json.loads(Path("configB_1m_results.json").read_text(encoding="utf-8"))

    pts = []  # (label, intensity, survival, ci, config)
    a_names = {"random": "Random", "policy_iteration": "Policy Iteration",
               "q_learning": "Q-Learning", "sarsa": "SARSA"}
    for k, lab in a_names.items():
        m = config_a[k]
        pts.append((lab, m["mean_intensity"], m["survival_rate"] * 100,
                    m.get("survival_ci", 0) * 100, "A"))
    b_names = {"random": "Random", "expert": "Expert", "bc": "BC",
               "DQN-v2 1M": "DQN", "PPO-v2 1M": "PPO", "A2C 1M": "A2C"}
    for k, lab in b_names.items():
        if k in config_b:
            m = config_b[k]["All"]
            pts.append((lab, m["intensity"], m["survival"] * 100,
                        m.get("survival_ci", 0) * 100, "B"))

    fig, ax = plt.subplots(figsize=(9, 6))
    for lab, x, y, ci, cfg in pts:
        color = "#2ecc71" if cfg == "A" else "#2980b9"
        marker = "^" if cfg == "A" else "o"
        ax.errorbar(x, y, yerr=ci, fmt=marker, color=color, ms=9, capsize=3, alpha=0.85)
        ax.annotate(f"{lab} ({cfg})", (x, y), textcoords="offset points",
                    xytext=(7, 4), fontsize=8)
    ax.set_xlabel("Mean treatment intensity per step  (lower = more parsimonious)")
    ax.set_ylabel("Survival rate (%)")
    ax.set_title("Treatment intensity vs survival - the lambda trade-off\n"
                 "(triangles = Config A discrete MDP, circles = Config B clinical env)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/configAB_pareto.png", dpi=150, bbox_inches="tight")
    print("Saved plots/configAB_pareto.png")


# --------------------------------------------------------------------------- #
# 3. DQN Q-value overestimation diagnostic
# --------------------------------------------------------------------------- #
def overestimation_diagnostic(tag="dqn_v2_1m", n_episodes=300):
    """Compare the DQN's predicted value of the greedy action at each visited state
    with the actual Monte Carlo return obtained from that state. A predicted value
    systematically above the realised return is the overestimation bias discussed in
    Lecture 5 (the motivation for Double DQN).
    """
    import torch
    import sepsis_rl as srl
    model = srl._load_model("DQN", tag, use_best=True)
    obs_rms = srl._load_obs_rms(tag)
    env = srl.make_clinical_env()
    rng = np.random.RandomState(SEED)

    pred_vals, mc_returns = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.randint(100_000)))
        traj = []
        done = False
        while not done:
            on = np.asarray(srl._normalize_obs(obs, obs_rms), dtype=np.float32)
            with torch.no_grad():
                q = model.policy.q_net(torch.as_tensor(on, device=model.device).unsqueeze(0))
            v = float(q.max().item())
            a = int(q.argmax().item())
            traj.append(v)
            obs, r, te, tr, _ = env.step(a)
            done = te or tr
            traj[-1] = (v, r)
        # discounted return-to-go (gamma = 1)
        g = 0.0
        for i in range(len(traj) - 1, -1, -1):
            v, r = traj[i]
            g = r + g
            pred_vals.append(v)
            mc_returns.append(g)
    env.close()

    pred_vals = np.array(pred_vals); mc_returns = np.array(mc_returns)
    bias = float(np.mean(pred_vals - mc_returns))
    print(f"[overestimation] mean predicted Q = {pred_vals.mean():.3f}, "
          f"mean realised return = {mc_returns.mean():.3f}, bias = {bias:+.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(mc_returns, pred_vals, s=8, alpha=0.25, color="#2980b9")
    lim = [min(mc_returns.min(), pred_vals.min()), max(mc_returns.max(), pred_vals.max())]
    ax.plot(lim, lim, ls="--", color="gray", label="predicted = realised")
    ax.set_xlabel("Realised Monte Carlo return")
    ax.set_ylabel("DQN predicted value (max_a Q)")
    ax.set_title(f"DQN value calibration / overestimation\n(mean bias = {bias:+.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/configB_overestimation.png", dpi=150, bbox_inches="tight")
    print("Saved plots/configB_overestimation.png")
    Path("configB_overestimation.json").write_text(
        json.dumps({"mean_predicted_q": float(pred_vals.mean()),
                    "mean_realised_return": float(mc_returns.mean()),
                    "bias": bias}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print("=== Config A with bootstrap CIs ===")
    ca = config_a_with_ci()
    print("\n=== Pareto plot ===")
    pareto_plot(ca)
    print("\n=== DQN overestimation diagnostic ===")
    overestimation_diagnostic()
    print("\nDone.")
