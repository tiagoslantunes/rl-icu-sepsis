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

# --------------------------------------------------------------------------- #
# Globals / setup
# --------------------------------------------------------------------------- #
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
LOGS_DIR = "logs"

ALGOS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}

# Sensible defaults per algorithm (close to the original notebook).
DEFAULT_HP = {
    "DQN": dict(policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=1e-4, buffer_size=100_000, learning_starts=1000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1000,
                exploration_fraction=0.20, exploration_final_eps=0.05),
    "PPO": dict(policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01),
    "A2C": dict(policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0, ent_coef=0.01),
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

    raw = {b: {"ret": [], "surv": []} for b in BUCKETS}
    env = make_clinical_env()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        ep_noisy = bool(info.get("noisy_episode", False))
        ep_missing = info.get("missing_features") is not None
        ep_acute = False
        done, ep_ret, last_r = False, 0.0, 0.0
        while not done:
            action, _ = model.predict(_normalize_obs(obs, obs_rms), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_ret += reward
            last_r = reward
            if info.get("acute_event", False):
                ep_acute = True
        surv = 1.0 if last_r > 0.5 else 0.0

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
    env.close()

    results: Dict[str, Dict[str, float]] = {}
    for b in BUCKETS:
        if raw[b]["ret"]:
            results[b] = {"return": float(np.mean(raw[b]["ret"])),
                          "survival": float(np.mean(raw[b]["surv"]))}
            print(f"[{tag}] {b:8s} n={len(raw[b]['ret']):4d}  "
                  f"return={results[b]['return']:.4f}  "
                  f"survival={results[b]['survival']:.1%}")
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
            slope  = coeffs[0] * 1e5   # por 100k passos
            ax.plot(ts, trend, ls="-", lw=2.0, color=line.get_color(), alpha=0.5,
                    label=f"{label} tendência ({slope:+.4f}/100k)")
            print(f"  [{label}] declive = {slope:+.5f} por 100k passos "
                  f"({'a subir' if slope > 0 else 'a descer'})")
        plotted += 1

    if baseline is not None:
        ax.axhline(baseline, color="gray", ls="--", lw=1.5, label="Random baseline")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Mean eval return (EMA α={smooth})")
    title = "Config B - Learning curves"
    if zoom_steps:
        title += f" (primeiros {zoom_steps:,} passos)"
    if show_trend:
        title += " + tendência linear"
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
    """Plota um subplot por algoritmo, lado a lado, com curva EMA e reta de
    tendência linear a cheio (mesma cor, alpha mais baixo).

    Args:
        tags_labels : tag -> label, um por algoritmo
        baseline    : linha horizontal do agente aleatório
        smooth      : EMA alpha
        zoom_steps  : limitar eixo x aos primeiros N passos
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

        # Reta de tendência — a cheio, mesma cor, alpha médio
        if len(ts) >= 2:
            coeffs = np.polyfit(ts, sm, 1)
            trend  = np.polyval(coeffs, ts)
            slope  = coeffs[0] * 1e5
            ax.plot(ts, trend, ls="-", lw=2.0, color="steelblue", alpha=0.45,
                    label=f"Tendência ({slope:+.4f}/100k)")
            direction = "a subir ↑" if slope > 0 else "a descer ↓"
            print(f"  [{label}] declive = {slope:+.5f}/100k passos  {direction}")

        if baseline is not None:
            ax.axhline(baseline, color="gray", ls="--", lw=1.2,
                       label="Random baseline")

        title = label
        if zoom_steps:
            title += f"\n(primeiros {zoom_steps:,} passos)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Timesteps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"Mean eval return (EMA α={smooth})")
    fig.suptitle("Config B — Curvas de aprendizagem por algoritmo", fontsize=13)
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
        cell = res.get(condition, res[list(res)[-1]])
        row = {"Agent": label,
               f"Return ({condition})": round(cell["return"], 4),
               f"Survival ({condition})": f"{cell['survival']:.1%}"}
        if random_ret:
            row["vs Random"] = f"{(cell['return'] / random_ret - 1) * 100:+.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (Optuna) - short proxy budget
# --------------------------------------------------------------------------- #
def _suggest_hp(trial, algo: str) -> dict:
    if algo == "DQN":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            buffer_size=trial.suggest_categorical("buffer_size", [50_000, 100_000]),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            gamma=trial.suggest_float("gamma", 0.95, 0.999),
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
            gamma=trial.suggest_float("gamma", 0.95, 0.999),
            gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        )
    # A2C
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        n_steps=trial.suggest_categorical("n_steps", [5, 8, 16]),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
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
    """Para cada algoritmo, carrega o eval log do melhor trial do Optuna e plota
    a curva com EMA + linha de tendência linear (declive visível).

    Args:
        algos_or_studies : pode ser:
            - dict algo -> optuna.Study  (quando tens os studies em memória)
            - list/tuple de nomes de algos  ex: ['DQN','PPO','A2C']
              (descobre o melhor trial directamente dos logs em disco,
               não precisa do kernel ter corrido o Optuna)
        baseline : return do agente aleatório
        smooth   : EMA alpha
        n_trials : número de trials que foram corridos (para a busca em disco)
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
            print(f"  (sem eval log para '{tag}')")
            continue

        if len(ts) < 3:
            print(f"  [{algo}] trial {best_t} tem apenas {len(ts)} ponto(s) de "
                  f"avaliação — não é possível traçar curva. "
                  f"Re-corre tune() para obter eval logs com múltiplos pontos.")
            continue

        sm = _ema(mean, smooth)
        line, = ax.plot(ts, sm, lw=2.5, label=f"{algo} (trial {best_t})")
        ax.plot(ts, mean, lw=0.8, alpha=0.15, color=line.get_color())

        # Linha de tendência linear
        coeffs = np.polyfit(ts, sm, 1)
        trend = np.polyval(coeffs, ts)
        slope = coeffs[0] * 1e5          # por 100k passos
        ax.plot(ts, trend, ls="--", lw=1.5, color=line.get_color(),
                label=f"{algo} tendência ({slope:+.4f}/100k steps)")
        print(f"[{algo}] melhor trial={best_t}  declive={slope:+.5f}/100k steps")
        plotted += 1

    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1.5, label="Random baseline")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Mean eval return (EMA α={smooth})")
    ax.set_title("Config B — Melhor trial Optuna por algoritmo + tendência linear")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show() if show else plt.close(fig)
    return path
