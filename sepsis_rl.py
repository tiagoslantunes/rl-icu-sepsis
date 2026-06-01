"""
sepsis_rl.py - All implementation code for Config B (continuous-STATE ICU-Sepsis).

The notebook should only *call* these functions: train agents, evaluate them
under clinical failure modes, and produce/save plots. Every figure is written
to the ``outputs/`` directory.

Note on the action space: Config B has a CONTINUOUS observation space (47-dim)
but the ACTION space is still Discrete(25) - same as Config A. We therefore use
the discrete-action SB3 algorithms below (value-based DQN, on-policy PPO/A2C).

Agents
------
DQN  - value-based, off-policy
PPO  - policy-gradient, on-policy, clipped objective
A2C  - policy-gradient, on-policy, unclipped (PPO ablation)

For each agent you can train a "v1" (no normalization) and a "v2"
(VecNormalize on observations). v2 changes *only* the normalization, giving a
clean ablation of "what does observation normalization buy us".
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.wrappers import make_clinical_env
from envs.env_setup import INTENSITY, N_ACTIONS

# --------------------------------------------------------------------------- #
# Globals / setup
# --------------------------------------------------------------------------- #
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
LOGS_DIR = "logs"

ALGOS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}

# Discount factor: 1.0 follows the ICU-Sepsis paper convention (finite, short
# episodes; the objective is survival probability) and MATCHES Config A, so the
# A-vs-B comparison is apples-to-apples. With ~10-step episodes there is no
# bootstrapping-stability reason to discount.
GAMMA = 1.0

# Sensible defaults per algorithm (close to the original notebook).
DEFAULT_HP = {
    "DQN": dict(policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=1e-4, buffer_size=100_000, learning_starts=1000,
                batch_size=64, gamma=GAMMA, train_freq=4, target_update_interval=1000,
                exploration_fraction=0.20, exploration_final_eps=0.05),
    "PPO": dict(policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
                gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01),
    "A2C": dict(policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=7e-4, n_steps=5, gamma=GAMMA, gae_lambda=1.0, ent_coef=0.01),
}

# Robustness buckets. The environment configuration is NEVER changed: every
# agent is evaluated on the default make_clinical_env() and each episode is
# assigned to buckets using the info flags the wrappers emit
# (noisy_episode / missing_features / acute_event) - the original methodology.
# "All" contains every episode.
BUCKETS = ["All", "Clean", "Noisy", "Missing", "Acute"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _monitored_env(**env_kwargs):
    return Monitor(make_clinical_env(**env_kwargs))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class _SyncVecNormCallback(BaseCallback):
    """Copy VecNormalize running stats from the training env to the eval env so
    the 'best model' is selected on correctly normalized observations."""

    def __init__(self, train_env, eval_env, eval_freq: int):
        super().__init__()
        self._train_env = train_env
        self._eval_env = eval_env
        self._eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self._eval_freq == 0 and hasattr(self._train_env, "obs_rms"):
            self._eval_env.obs_rms = self._train_env.obs_rms
        return True


def train_agent(
    algo: str,
    timesteps: int = 150_000,
    normalize: bool = False,
    seed: int = SEED,
    hyperparams: Optional[dict] = None,
    tag: Optional[str] = None,
    eval_freq: int = 5000,
    n_eval_episodes: int = 50,
    verbose: int = 0,
    progress_bar: bool = True,
) -> Tuple[object, str]:
    """Train one agent; save best model (+ VecNormalize stats + eval log).

    Returns ``(model, tag)``. ``tag`` is the sub-directory name used under
    ``models/`` and ``logs/`` and identifies the run in later plots.
    """
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo {algo!r}; choose from {list(ALGOS)}")
    set_seed(seed)
    cls = ALGOS[algo]
    tag = tag or f"{algo.lower()}{'_v2' if normalize else ''}"
    model_dir = os.path.join(MODELS_DIR, tag)
    log_dir = os.path.join(LOGS_DIR, tag)
    ensure_dirs(model_dir, log_dir)

    hp = {**DEFAULT_HP[algo], **(hyperparams or {})}

    if normalize:
        train_env = VecNormalize(DummyVecEnv([_monitored_env]),
                                 norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(DummyVecEnv([_monitored_env]),
                                norm_obs=True, norm_reward=False, clip_obs=10.0,
                                training=False)
    else:
        train_env = DummyVecEnv([_monitored_env])
        eval_env = DummyVecEnv([_monitored_env])

    model = cls("MlpPolicy", train_env, seed=seed, device=DEVICE,
                verbose=verbose, tensorboard_log=log_dir, **hp)

    callbacks: List[BaseCallback] = [EvalCallback(
        eval_env, best_model_save_path=model_dir, log_path=log_dir,
        eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        deterministic=True, render=False, verbose=0,
    )]
    if normalize:
        callbacks.append(_SyncVecNormCallback(train_env, eval_env, eval_freq))

    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=progress_bar)
    model.save(os.path.join(model_dir, f"{tag}_final"))
    if normalize:
        train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()
    print(f"[{tag}] training complete ({timesteps:,} steps).")
    return model, tag


# --------------------------------------------------------------------------- #
# Evaluation (robustness under clinical failure modes)
# --------------------------------------------------------------------------- #
def _load_obs_rms(tag: str):
    """Load VecNormalize obs statistics for a run, or None if not normalized."""
    path = os.path.join(MODELS_DIR, tag, "vecnormalize.pkl")
    if not os.path.exists(path):
        return None
    vn = VecNormalize.load(path, DummyVecEnv([_monitored_env]))
    return vn.obs_rms


def _normalize_obs(obs, obs_rms, clip: float = 10.0):
    if obs_rms is None:
        return obs
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -clip, clip)


def _load_model(algo: str, tag: str, use_best: bool):
    """Load best_model if available & valid, otherwise the final checkpoint."""
    cls = ALGOS[algo]
    best = os.path.join(MODELS_DIR, tag, "best_model")
    final = os.path.join(MODELS_DIR, tag, f"{tag}_final")
    if use_best and os.path.exists(best + ".zip"):
        try:
            return cls.load(best, device=DEVICE)
        except Exception as exc:
            print(f"  (best_model for '{tag}' unreadable: {exc}; using final)")
    return cls.load(final, device=DEVICE)


def evaluate_conditions(
    tag: str,
    algo: str,
    n_episodes: int = 300,
    use_best: bool = True,
    seed_offset: int = 20_000,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained agent on the DEFAULT clinical env, bucketing episodes
    by failure mode.

    The environment configuration is never changed: we run ``make_clinical_env()``
    with its default parameters and split episodes using the wrappers' info
    flags (noisy_episode / missing_features / acute_event) - the same
    methodology as the original notebook. The only fix vs the original is that
    survival is read from the terminal reward (death -> 0, survival -> ~1)
    instead of a fragile total-return threshold.

    Returns ``{bucket: {"return": mean, "survival": rate}}`` for buckets that
    actually occurred ("All" is always present).
    """
    model = _load_model(algo, tag, use_best)
    obs_rms = _load_obs_rms(tag)

    raw = {b: {"ret": [], "surv": [], "intens": []} for b in BUCKETS}
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)  # for the dose-grid heatmap
    env = make_clinical_env()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        ep_noisy = bool(info.get("noisy_episode", False))
        ep_missing = info.get("missing_features") is not None
        ep_acute = False
        done, ep_ret, last_r = False, 0.0, 0.0
        ep_intens, ep_steps = 0.0, 0
        while not done:
            action, _ = model.predict(_normalize_obs(obs, obs_rms), deterministic=True)
            a = int(action)
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            ep_ret += reward
            last_r = reward
            ep_intens += INTENSITY[a]
            action_counts[a] += 1
            ep_steps += 1
            if info.get("acute_event", False):
                ep_acute = True
        surv = 1.0 if last_r > 0.5 else 0.0
        intens = ep_intens / max(ep_steps, 1)            # mean treatment intensity / step

        buckets_hit = ["All"]
        if not ep_noisy and not ep_missing and not ep_acute:
            buckets_hit.append("Clean")
        if ep_noisy:
            buckets_hit.append("Noisy")
        if ep_missing:
            buckets_hit.append("Missing")
        if ep_acute:
            buckets_hit.append("Acute")
        for b in buckets_hit:
            raw[b]["ret"].append(ep_ret)
            raw[b]["surv"].append(surv)
            raw[b]["intens"].append(intens)
    env.close()

    results: Dict[str, Dict[str, float]] = {}
    for b in BUCKETS:
        if raw[b]["ret"]:
            results[b] = {"return": float(np.mean(raw[b]["ret"])),
                          "survival": float(np.mean(raw[b]["surv"])),
                          "intensity": float(np.mean(raw[b]["intens"]))}
            print(f"[{tag}] {b:8s} n={len(raw[b]['ret']):4d}  "
                  f"return={results[b]['return']:.4f}  "
                  f"survival={results[b]['survival']:.1%}  "
                  f"intensity={results[b]['intensity']:.3f}")
    # Overall action distribution (not a bucket; filtered out by BUCKETS-based plots).
    results["__actions__"] = action_counts.tolist()
    return results


def random_baseline(n_episodes: int = 1000, seed: int = SEED) -> Dict[str, float]:
    """Random-policy baseline on the full clinical env (the target to beat)."""
    env = make_clinical_env()
    returns, survivals = [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done, last_r, ep_ret = False, 0.0, 0.0
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            ep_ret += reward
            last_r = reward
        returns.append(ep_ret)
        survivals.append(1.0 if last_r > 0.5 else 0.0)
    env.close()
    out = {"return": float(np.mean(returns)), "survival": float(np.mean(survivals))}
    print(f"[random] return={out['return']:.4f}  survival={out['survival']:.1%}")
    return out


# --------------------------------------------------------------------------- #
# Creative extension: clinical interpretability (feature importance)
# --------------------------------------------------------------------------- #
def feature_importance_dqn(
    tag: str,
    n_states: int = 500,
    seed: int = SEED,
    top_n: int = 20,
    filename: str = "configB_feature_importance.png",
    show: bool = True,
):
    """Permutation feature importance for a trained DQN, on its Q-network.

    For each of the 47 clinical features we permute its values across a sample of
    states and measure the mean |Delta Q| on the originally-chosen action. Large
    values = the agent's decision relies heavily on that feature. If the run used
    VecNormalize, the saved obs stats are applied first so |Delta Q| is computed on
    the same normalized inputs the network was trained on.

    Returns ``(importances, order)`` and saves a horizontal bar chart highlighting
    established sepsis-severity markers.
    """
    import torch
    import matplotlib.pyplot as plt
    from envs.continuous_sepsis_env import FEATURE_NAMES

    ensure_dirs(OUTPUT_DIR)
    model = _load_model("DQN", tag, use_best=True)
    obs_rms = _load_obs_rms(tag)

    # Sample patient states by rolling out a random policy on the clinical env.
    rng = np.random.RandomState(seed)
    env = make_clinical_env()
    states = []
    while len(states) < n_states:
        obs, _ = env.reset(seed=int(rng.randint(100_000)))
        states.append(obs.copy())
        done = False
        while not done and len(states) < n_states:
            obs, _, te, tr, _ = env.step(env.action_space.sample())
            states.append(obs.copy())
            done = te or tr
    env.close()
    states = np.asarray(states[:n_states], dtype=np.float32)

    def qvals(arr):
        t = torch.as_tensor(np.asarray(_normalize_obs(arr, obs_rms), dtype=np.float32),
                            device=model.device)
        with torch.no_grad():
            return model.policy.q_net(t).cpu().numpy()

    base_q = qvals(states)
    chosen = base_q.argmax(axis=1)
    base_chosen = base_q[np.arange(len(states)), chosen]

    importances = np.zeros(len(FEATURE_NAMES))
    for j in range(len(FEATURE_NAMES)):
        perturbed = states.copy()
        perturbed[:, j] = rng.permutation(perturbed[:, j])
        pq = qvals(perturbed)[np.arange(len(states)), chosen]
        importances[j] = float(np.mean(np.abs(base_chosen - pq)))

    order = np.argsort(importances)[::-1]
    markers = {"SOFA", "Lactate", "Mean_BP", "Systolic_BP", "Diastolic_BP",
               "Urine_Output", "Creatinine", "HR", "RespRate", "GCS", "FiO2"}
    labels = [FEATURE_NAMES[i] for i in order[:top_n]]
    vals = importances[order[:top_n]]
    colors = ["#E53935" if l in markers else "#42A5F5" for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n)[::-1], vals, color=colors, alpha=0.85)
    ax.set_yticks(range(top_n)[::-1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |Delta Q| when feature permuted")
    ax.set_title(f"DQN feature importance - top {top_n} clinical variables\n"
                 "(red = established sepsis-severity markers)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#E53935", label="Sepsis severity marker"),
                       Patch(facecolor="#42A5F5", label="Other clinical variable")],
              loc="lower right")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    print("Top 10 decision-relevant features:")
    for r, idx in enumerate(order[:10], 1):
        mk = " <- sepsis marker" if FEATURE_NAMES[idx] in markers else ""
        print(f"  {r:2d}. {FEATURE_NAMES[idx]:<16s} {importances[idx]:.5f}{mk}")
    plt.show() if show else plt.close(fig)
    return importances, order


# --------------------------------------------------------------------------- #
# Cross-config comparison (reads the REAL Config A metrics from JSON)
# --------------------------------------------------------------------------- #
def compare_configs(
    results_b: Dict[str, Dict[str, object]],
    configA_path: str = "configA_results.json",
    condition: str = "All",
):
    """Build an honest Config A vs Config B table.

    Config A numbers are read from ``configA_results.json`` (produced by the
    Config A notebook) - never hard-coded. Config B numbers come from
    ``evaluate_conditions`` results passed in ``results_b``.
    """
    import json
    import pandas as pd

    rows = []
    if os.path.exists(configA_path):
        with open(configA_path) as f:
            a = json.load(f)
        nice = {"random": "Random", "policy_iteration": "Policy Iteration",
                "q_learning": "Q-Learning", "sarsa": "SARSA"}
        for key, label in nice.items():
            if key in a:
                m = a[key]
                rows.append({"Config": "A", "Agent": label,
                             "Return": round(m["mean_return"], 4),
                             "Survival": f"{m['survival_rate']*100:.1f}%",
                             "Intensity": round(m["mean_intensity"], 3)})
    else:
        print(f"  ({configA_path} not found - run the Config A notebook first)")

    for label, res in results_b.items():
        cell = res.get(condition)
        if cell is None:
            continue
        rows.append({"Config": "B", "Agent": label,
                     "Return": round(cell["return"], 4),
                     "Survival": f"{cell['survival']*100:.1f}%",
                     "Intensity": round(cell.get("intensity", float("nan")), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting (all saved to OUTPUT_DIR)
# --------------------------------------------------------------------------- #
def _ema(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return np.array(out)


def _load_eval_log(tag: str):
    path = os.path.join(LOGS_DIR, tag, "evaluations.npz")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path)
    results = data["results"]
    return data["timesteps"], results.mean(axis=1), results.std(axis=1)



def plot_learning_curves(
    tags_labels: Dict[str, str],
    baseline: Optional[float] = None,
    filename: str = "configB_learning_curves.png",
    smooth: float = 0.6,
    zoom_steps: Optional[int] = None,
    show_trend: bool = False,
    show: bool = True,
):
    """Plot EMA-smoothed eval-return curves for one run per algorithm.

    Args:
        tags_labels : run tag -> legend label
        baseline    : horizontal dashed line (e.g. random baseline return)
        smooth      : EMA alpha in [0,1]. 0 = raw, 1 = fully flat.
        zoom_steps  : if set, only show the first N timesteps.
        show_trend  : if True, draws a linear trend line (regressão linear)
                      and prints the slope per 100k steps.
    """
    import matplotlib.pyplot as plt

    ensure_dirs(OUTPUT_DIR)
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for tag, label in tags_labels.items():
        ts, mean, std = _load_eval_log(tag)
        if ts is None:
            print(f"  (no eval log for '{tag}' - run training first)")
            continue
        if zoom_steps is not None:
            mask = ts <= zoom_steps
            ts, mean, std = ts[mask], mean[mask], std[mask]
        if len(ts) == 0:
            continue
        sm = _ema(mean, smooth)
        line, = ax.plot(ts, sm, lw=2.5, label=label)
        ax.plot(ts, mean, lw=0.8, alpha=0.15, color=line.get_color())

        if show_trend and len(ts) >= 2:
            coeffs = np.polyfit(ts, sm, 1)
            trend  = np.polyval(coeffs, ts)
            slope  = coeffs[0] * 1e5   # per 100k steps
            ax.plot(ts, trend, ls="-", lw=2.0, color=line.get_color(), alpha=0.5,
                    label=f"{label} trend ({slope:+.4f}/100k)")
            print(f"  [{label}] slope = {slope:+.5f} per 100k steps "
                  f"({'rising' if slope > 0 else 'falling'})")
        plotted += 1

    if baseline is not None:
        ax.axhline(baseline, color="gray", ls="--", lw=1.5, label="Random baseline")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Mean eval return (EMA alpha={smooth})")
    title = "Config B - Learning curves"
    if zoom_steps:
        title += f" (first {zoom_steps:,} steps)"
    if show_trend:
        title += " + linear trend"
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show() if show else plt.close(fig)
    return path


def plot_curves_grid(
    tags_labels: Dict[str, str],
    baseline: Optional[float] = None,
    filename: str = "configB_curves_grid.png",
    smooth: float = 0.6,
    zoom_steps: Optional[int] = None,
    show: bool = True,
):
    """One subplot per algorithm, side by side, with an EMA curve and a solid
    linear-trend line (same colour, lower alpha).

    Args:
        tags_labels : tag -> label, one per algorithm
        baseline    : horizontal line for the random agent
        smooth      : EMA alpha
        zoom_steps  : limit the x-axis to the first N steps
    """
    import matplotlib.pyplot as plt

    ensure_dirs(OUTPUT_DIR)
    n = len(tags_labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (tag, label) in zip(axes, tags_labels.items()):
        ts, mean, _ = _load_eval_log(tag)
        if ts is None:
            ax.set_title(f"{label}\n(sem log)")
            continue
        if zoom_steps is not None:
            mask = ts <= zoom_steps
            ts, mean = ts[mask], mean[mask]
        if len(ts) == 0:
            continue

        sm = _ema(mean, smooth)

        # Curva raw (muito transparente)
        ax.plot(ts, mean, lw=0.8, alpha=0.15, color="steelblue")
        # Curva EMA
        ax.plot(ts, sm, lw=2.5, color="steelblue", label="EMA")

        # Trend line - solid, same colour, medium alpha
        if len(ts) >= 2:
            coeffs = np.polyfit(ts, sm, 1)
            trend  = np.polyval(coeffs, ts)
            slope  = coeffs[0] * 1e5
            ax.plot(ts, trend, ls="-", lw=2.0, color="steelblue", alpha=0.45,
                    label=f"Trend ({slope:+.4f}/100k)")
            direction = "rising" if slope > 0 else "falling"
            print(f"  [{label}] slope = {slope:+.5f}/100k steps  {direction}")

        if baseline is not None:
            ax.axhline(baseline, color="gray", ls="--", lw=1.2,
                       label="Random baseline")

        title = label
        if zoom_steps:
            title += f"\n(first {zoom_steps:,} steps)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Timesteps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"Mean eval return (EMA alpha={smooth})")
    fig.suptitle("Config B - Learning curves per algorithm", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show() if show else plt.close(fig)
    return path


def plot_robustness(
    results_by_agent: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "return",
    baseline: Optional[float] = None,
    filename: str = "configB_robustness.png",
    show: bool = True,
):
    """Grouped bar chart of a metric per failure mode, from *real* eval results.

    ``results_by_agent`` maps label -> output of ``evaluate_conditions``.
    """
    import matplotlib.pyplot as plt

    ensure_dirs(OUTPUT_DIR)
    agents = list(results_by_agent)
    present = {c for r in results_by_agent.values() for c in r}
    conditions = [b for b in BUCKETS if b in present]
    x = np.arange(len(conditions))
    width = 0.8 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(agents):
        vals = [results_by_agent[label].get(c, {}).get(metric, np.nan) for c in conditions]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label)
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls="--", label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Survival rate" if metric == "survival" else "Mean return")
    ax.set_title(f"Config B - Robustness by failure mode ({metric})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show() if show else plt.close(fig)
    return path


def results_table(
    results_by_agent: Dict[str, Dict[str, Dict[str, float]]],
    random_ret: Optional[float] = None,
    condition: str = "All",
):
    """Return a tidy pandas DataFrame summarizing performance on one condition."""
    import pandas as pd

    rows = []
    for label, res in results_by_agent.items():
        cell = res.get(condition)
        if cell is None:
            continue
        row = {"Agent": label,
               f"Return ({condition})": round(cell["return"], 4),
               f"Survival ({condition})": f"{cell['survival']:.1%}",
               f"Intensity ({condition})": round(cell.get("intensity", float("nan")), 3)}
        if random_ret:
            row["vs Random"] = f"{(cell['return'] / random_ret - 1) * 100:+.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_dose_grid(
    results_by_agent: Dict[str, Dict[str, object]],
    filename: str = "configB_dose_grid.png",
    show: bool = True,
):
    """5x5 vasopressor x IV-fluid dose heatmaps (one per agent), from the action
    distribution collected in ``evaluate_conditions`` (the ``__actions__`` entry).

    Reveals each learned policy's treatment philosophy (conservative vs aggressive),
    which is the clinical-interpretability counterpart of the Config A action plots.
    """
    import matplotlib.pyplot as plt

    ensure_dirs(OUTPUT_DIR)
    agents = [a for a in results_by_agent if "__actions__" in results_by_agent[a]]
    if not agents:
        print("  (no '__actions__' in results - re-run evaluate_conditions)")
        return None
    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    levels = ["None", "Low", "Med", "High", "V.High"]
    for ax, label in zip(axes, agents):
        counts = np.asarray(results_by_agent[label]["__actions__"], dtype=float)
        grid = counts.reshape(5, 5)                  # rows = vaso, cols = fluid
        grid = grid / grid.sum() if grid.sum() else grid
        im = ax.imshow(grid, cmap="Blues", vmin=0)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{grid[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_xticks(range(5)); ax.set_xticklabels(levels, rotation=45, fontsize=8)
        ax.set_yticks(range(5)); ax.set_yticklabels(levels, fontsize=8)
        ax.set_xlabel("IV fluid"); ax.set_ylabel("Vasopressor")
        ax.set_title(f"{label}\n(% of steps)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Config B - Learned dose distribution (5x5 action grid)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show() if show else plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (Optuna) - short proxy budget
# --------------------------------------------------------------------------- #
def _suggest_hp(trial, algo: str) -> dict:
    # gamma is fixed at GAMMA (=1.0, env convention) and intentionally NOT tuned,
    # so it stays consistent with Config A and across all trials.
    if algo == "DQN":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            buffer_size=trial.suggest_categorical("buffer_size", [50_000, 100_000]),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            train_freq=trial.suggest_categorical("train_freq", [1, 4, 8]),
            exploration_fraction=trial.suggest_float("exploration_fraction", 0.1, 0.4),
            target_update_interval=trial.suggest_categorical(
                "target_update_interval", [500, 1000, 2000]),
        )
    if algo == "PPO":
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            n_steps=n_steps,
            batch_size=trial.suggest_categorical("batch_size", [64, 128]),
            gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        )
    # A2C
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        n_steps=trial.suggest_categorical("n_steps", [5, 8, 16]),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 1.0),
        ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
    )


def tune(
    algo: str,
    n_trials: int = 15,
    timesteps: int = 50_000,
    normalize: bool = True,
    n_eval_episodes: int = 50,
    seed: int = SEED,
):
    """Optuna study tuning ``algo`` with a short proxy budget. Returns the study.

    The objective trains a short run, then evaluates mean return on the full
    'All-failures' condition.
    """
    import optuna

    def objective(trial):
        hp = _suggest_hp(trial, algo)
        # eval_freq = 10% dos timesteps → 10 pontos por trial (curva visível)
        trial_eval_freq = max(1000, timesteps // 10)
        _, tag = train_agent(algo, timesteps=timesteps, normalize=normalize,
                             seed=seed, hyperparams=hp,
                             tag=f"{algo.lower()}_optuna_t{trial.number}",
                             eval_freq=trial_eval_freq, n_eval_episodes=5,
                             progress_bar=False)
        res = evaluate_conditions(tag, algo, n_episodes=n_eval_episodes)
        return res["All"]["return"]

    study = optuna.create_study(direction="maximize",
                                study_name=f"{algo}_configB")
    study.optimize(objective, n_trials=n_trials)
    print(f"\n[{algo}] Best value: {study.best_value:.4f}")
    print(f"[{algo}] Best params: {study.best_params}")
    return study


def save_optuna_plots(study, name: str, show: bool = True):
    """Save Optuna optimization-history and param-importance plots to outputs/."""
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as ov

    ensure_dirs(OUTPUT_DIR)
    paths = []
    for kind, fn in [("history", ov.plot_optimization_history),
                     ("importance", ov.plot_param_importances)]:
        try:
            ax = fn(study)
            fig = ax.figure
            fig.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"optuna_{name.lower()}_{kind}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            paths.append(path)
            print(f"Saved {path}")
            plt.show() if show else plt.close(fig)
        except Exception as exc:
            print(f"Could not plot {kind} for {name}: {exc}")
    return paths


def _find_best_optuna_trial(algo: str, n_trials: int = 15) -> Tuple[int, str]:
    """Lê os eval logs em disco e devolve (best_trial_number, tag)."""
    best_ret, best_t = -np.inf, 0
    for t in range(n_trials):
        tag = f"{algo.lower()}_optuna_t{t}"
        ts, mean, _ = _load_eval_log(tag)
        if mean is not None and mean.mean() > best_ret:
            best_ret, best_t = mean.mean(), t
    return best_t, f"{algo.lower()}_optuna_t{best_t}"


def plot_best_trial_curves(
    algos_or_studies,
    baseline: Optional[float] = None,
    smooth: float = 0.6,
    n_trials: int = 15,
    filename: str = "configB_optuna_best_curves.png",
    show: bool = True,
):
    """For each algorithm, load the eval log of the best Optuna trial and plot the
    curve with EMA + a linear-trend line (visible slope).

    Args:
        algos_or_studies : either
            - dict algo -> optuna.Study  (when the studies are in memory)
            - list/tuple of algo names, e.g. ['DQN','PPO','A2C']
              (finds the best trial directly from the on-disk logs, so the
               kernel need not have run Optuna)
        baseline : random-agent return
        smooth   : EMA alpha
        n_trials : number of trials run (for the on-disk search)
    """
    import matplotlib.pyplot as plt

    ensure_dirs(OUTPUT_DIR)
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0

    # Normalizar input: aceita studies dict ou lista de nomes
    if isinstance(algos_or_studies, dict):
        items = []
        for algo, study in algos_or_studies.items():
            best_t = study.best_trial.number
            items.append((algo, best_t, f"{algo.lower()}_optuna_t{best_t}"))
    else:
        items = []
        for algo in algos_or_studies:
            best_t, tag = _find_best_optuna_trial(algo, n_trials)
            items.append((algo, best_t, tag))

    for algo, best_t, tag in items:
        ts, mean, _ = _load_eval_log(tag)
        if ts is None:
            print(f"  (no eval log for '{tag}')")
            continue

        if len(ts) < 3:
            print(f"  [{algo}] trial {best_t} has only {len(ts)} evaluation "
                  f"point(s) - cannot draw a curve. "
                  f"Re-run tune() to get eval logs with multiple points.")
            continue

        sm = _ema(mean, smooth)
        line, = ax.plot(ts, sm, lw=2.5, label=f"{algo} (trial {best_t})")
        ax.plot(ts, mean, lw=0.8, alpha=0.15, color=line.get_color())

        # Linear trend line
        coeffs = np.polyfit(ts, sm, 1)
        trend = np.polyval(coeffs, ts)
        slope = coeffs[0] * 1e5          # per 100k steps
        ax.plot(ts, trend, ls="--", lw=1.5, color=line.get_color(),
                label=f"{algo} trend ({slope:+.4f}/100k steps)")
        print(f"[{algo}] best trial={best_t}  slope={slope:+.5f}/100k steps")
        plotted += 1

    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1.5, label="Random baseline")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Mean eval return (EMA alpha={smooth})")
    ax.set_title("Config B - Best Optuna trial per algorithm + linear trend")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show() if show else plt.close(fig)
    return path
