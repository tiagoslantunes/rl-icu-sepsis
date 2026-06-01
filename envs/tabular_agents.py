"""
tabular_agents.py — Config A Tabular RL Agents
===============================================
Three algorithms for the discrete ICU-Sepsis MDP (716 states, 25 actions).

Contents
--------
  PolicyIteration   – model-based (requires P and R matrices)
  QLearning         – model-free, off-policy TD control
  SARSA             – model-free, on-policy TD control
  evaluate_policy() – shared evaluation helper (survival rate + mean return)

All agents expose a consistent interface:
  agent.train(...)     → trains the agent and returns a learning history dict
  agent.get_policy()   → returns a (N_STATES,) int array of greedy actions
  agent.get_Q()        → returns the (N_STATES, N_ACTIONS) value matrix

References
----------
  Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed.
    - Policy Iteration:  Chapter 4.3
    - Q-Learning:        Chapter 6.5
    - SARSA:             Chapter 6.4
"""

import numpy as np
from envs.env_setup import N_STATES, N_ACTIONS, GAMMA, INTENSITY, make_sepsis_env


# ─────────────────────────────────────────────────────────────
#  Shared evaluation helper
# ─────────────────────────────────────────────────────────────

def evaluate_policy(policy: np.ndarray,
                    n_episodes: int = 1_000,
                    seed: int = 42) -> dict:
    """
    Roll out a deterministic policy and return performance metrics.

    Parameters
    ----------
    policy     : (N_STATES,) int array — action index per state.
    n_episodes : number of evaluation episodes.
    seed       : master RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
      'mean_return'      : float  – average episodic return.
      'survival_rate'    : float  – fraction of episodes that survive (return > 0).
      'mean_ep_length'   : float  – average steps per episode.
      'mean_intensity'   : float  – average treatment intensity per step.
      'returns'          : ndarray of per-episode returns.
    """
    np.random.seed(seed)
    env = make_sepsis_env()

    returns, lengths, intensities, survivals = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        total_r, steps, total_intensity, last_r, done = 0.0, 0, 0.0, 0.0, False
        while not done:
            action = int(policy[int(obs)])
            obs, r, te, tr, _ = env.step(action)
            total_r += r
            last_r = r                       # terminal step carries the survival signal
            total_intensity += INTENSITY[action]
            steps += 1
            done = te or tr
        returns.append(total_r)
        lengths.append(steps)
        intensities.append(total_intensity / max(steps, 1))
        # Survival is read from the terminal reward (survival -> ~+1, death -> ~0),
        # which is robust to the accumulated intensity penalty. This MATCHES the
        # Config B definition (sepsis_rl.evaluate_conditions) so the two configs
        # are directly comparable.
        survivals.append(1.0 if last_r > 0.5 else 0.0)

    env.close()
    returns = np.array(returns)
    return {
        'mean_return':    float(np.mean(returns)),
        'survival_rate':  float(np.mean(survivals)),
        'mean_ep_length': float(np.mean(lengths)),
        'mean_intensity': float(np.mean(intensities)),
        'returns':        returns,
    }


# ─────────────────────────────────────────────────────────────
#  1. Policy Iteration  (model-based)
# ─────────────────────────────────────────────────────────────

class PolicyIteration:
    """
    Policy Iteration for a known tabular MDP.

    Requires access to the full transition matrix P and expected reward matrix R,
    which are available in ICU-Sepsis-v2 via env.unwrapped._tx_mat and _r_mat.

    Algorithm (Sutton & Barto §4.3)
    --------------------------------
    Alternate between:
      Policy Evaluation  — solve V_π iteratively until |ΔV| < θ.
      Policy Improvement — update π to be greedy w.r.t. V_π.
    Repeat until the policy is stable.

    Parameters
    ----------
    P     : (S, A, S') transition probability tensor.
    R     : (S, A)     expected reward matrix  E[r | s, a].
    gamma : discount factor (default 1.0, as per ICU-Sepsis paper).
    theta : convergence threshold for policy evaluation (default 1e-8).
    """

    def __init__(self,
                 P: np.ndarray,
                 R: np.ndarray,
                 gamma: float = GAMMA,
                 theta: float = 1e-8):
        self.P     = P          # (S, A, S')
        self.R     = R          # (S, A)
        self.gamma = gamma
        self.theta = theta
        self.S, self.A = R.shape

        # Initialise a random policy and zero value function
        self._policy = np.zeros(self.S, dtype=int)
        self._V      = np.zeros(self.S)

    # ----------------------------------------------------------
    def _policy_evaluation(self) -> int:
        """Iterative policy evaluation. Returns number of sweeps."""
        sweeps = 0
        while True:
            delta = 0.0
            for s in range(self.S):
                a   = self._policy[s]
                # V(s) = Σ_{s'} P(s'|s,a) * [R(s,a) + γ * V(s')]
                v   = float(np.dot(self.P[s, a], self.R[s, a] + self.gamma * self._V))
                delta = max(delta, abs(v - self._V[s]))
                self._V[s] = v
            sweeps += 1
            if delta < self.theta:
                break
        return sweeps

    def _policy_evaluation_vectorised(self) -> int:
        """Vectorised policy evaluation (faster for large S). Returns sweeps."""
        sweeps = 0
        S, A = self.S, self.A
        # Index-select P and R by current policy
        while True:
            a_idx = self._policy                     # (S,)
            P_pi  = self.P[np.arange(S), a_idx, :]  # (S, S')
            R_pi  = self.R[np.arange(S), a_idx]      # (S,)
            V_new = R_pi + self.gamma * (P_pi @ self._V)
            delta = np.max(np.abs(V_new - self._V))
            self._V = V_new
            sweeps += 1
            if delta < self.theta:
                break
        return sweeps

    # ----------------------------------------------------------
    def _policy_improvement(self) -> bool:
        """Greedy policy improvement. Returns True if policy changed."""
        # Q(s, a) = Σ_{s'} P(s'|s,a) * [R(s,a) + γ * V(s')]
        Q = self.R + self.gamma * (self.P @ self._V)   # (S, A)
        new_policy = np.argmax(Q, axis=1)
        stable = np.all(new_policy == self._policy)
        self._policy = new_policy
        return not stable  # True if the policy changed (not yet stable)

    # ----------------------------------------------------------
    def train(self, max_iterations: int = 200, verbose: bool = True) -> dict:
        """
        Run Policy Iteration until convergence or max_iterations.

        Returns
        -------
        dict with:
          'iterations'     : total PI iterations run.
          'eval_sweeps'    : list of sweeps per evaluation phase.
          'policy_stable'  : bool — converged before hitting max_iterations.
        """
        history = {'iterations': 0, 'eval_sweeps': [], 'policy_stable': False}

        for i in range(max_iterations):
            # --- Policy Evaluation ---
            sweeps = self._policy_evaluation_vectorised()
            history['eval_sweeps'].append(sweeps)

            # --- Policy Improvement ---
            changed = self._policy_improvement()
            history['iterations'] = i + 1

            if verbose:
                print(f"  PI iter {i+1:3d} | eval sweeps: {sweeps:3d} | "
                      f"policy changed: {changed}")

            if not changed:          # policy is stable → converged
                history['policy_stable'] = True
                break

        if verbose:
            status = 'converged' if history['policy_stable'] else 'max iterations reached'
            print(f"\nPolicy Iteration {status} after "
                  f"{history['iterations']} iterations.")

        return history

    def get_policy(self) -> np.ndarray:
        """Return greedy policy: (N_STATES,) int array."""
        return self._policy.copy()

    def get_V(self) -> np.ndarray:
        """Return state-value function: (N_STATES,) float array."""
        return self._V.copy()

    def get_Q(self) -> np.ndarray:
        """Return action-value matrix: (N_STATES, N_ACTIONS) float array."""
        return self.R + self.gamma * (self.P @ self._V)


# ─────────────────────────────────────────────────────────────
#  2. Q-Learning  (model-free, off-policy)
# ─────────────────────────────────────────────────────────────

class QLearning:
    """
    Q-Learning (Watkins, 1989) — off-policy TD control.

    Updates the action-value function toward the greedy (max-Q) bootstrap target:
        Q(s, a) ← Q(s, a) + α [r + γ * max_{a'} Q(s', a') − Q(s, a)]

    Exploration uses an ε-greedy policy that linearly anneals from
    `epsilon_start` to `epsilon_end` over `epsilon_decay_steps` steps.

    Parameters
    ----------
    n_states       : number of discrete states  (default: N_STATES = 716).
    n_actions      : number of discrete actions (default: N_ACTIONS = 25).
    alpha          : learning rate              (default 0.1).
    gamma          : discount factor            (default GAMMA = 1.0).
    epsilon_start  : initial exploration rate   (default 1.0).
    epsilon_end    : final exploration rate     (default 0.05).
    epsilon_decay_steps : episodes over which ε decays linearly (default 5_000).
    seed           : RNG seed for reproducibility (default 42).
    """

    def __init__(self,
                 n_states: int  = N_STATES,
                 n_actions: int = N_ACTIONS,
                 alpha: float   = 0.1,
                 gamma: float   = GAMMA,
                 epsilon_start: float        = 1.0,
                 epsilon_end:   float        = 0.05,
                 epsilon_decay_steps: int    = 5_000,
                 seed: int = 42):

        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = epsilon_start
        self.eps_end   = epsilon_end
        self.eps_decay = epsilon_decay_steps
        self.seed      = seed

        np.random.seed(seed)
        self.Q = np.zeros((n_states, n_actions))

    # ----------------------------------------------------------
    def _epsilon(self, episode: int) -> float:
        """Linear ε schedule."""
        frac = min(episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _select_action(self, state: int, epsilon: float) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    # ----------------------------------------------------------
    def train(self,
              n_episodes: int = 10_000,
              eval_every:  int = 500,
              eval_eps:    int = 300,
              verbose:     bool = True) -> dict:
        """
        Train the Q-Learning agent.

        Parameters
        ----------
        n_episodes : total training episodes.
        eval_every : evaluate the greedy policy every this many episodes.
        eval_eps   : episodes used in each evaluation.
        verbose    : print progress.

        Returns
        -------
        dict with keys:
          'episode_returns'  : (n_episodes,) training episode returns.
          'eval_steps'       : episode indices where evaluation was run.
          'eval_returns'     : mean greedy return at each eval checkpoint.
          'eval_survival'    : survival rate at each eval checkpoint.
          'epsilons'         : ε value at each training episode.
        """
        env = make_sepsis_env()

        ep_returns  = []
        eval_steps, eval_returns, eval_survival = [], [], []
        epsilons    = []

        for ep in range(n_episodes):
            eps = self._epsilon(ep)
            epsilons.append(eps)

            obs, _ = env.reset(seed=np.random.randint(100_000))
            s = int(obs)
            total_r, done = 0.0, False

            while not done:
                a = self._select_action(s, eps)
                obs_next, r, te, tr, _ = env.step(a)
                s_next = int(obs_next)
                done   = te or tr

                # Q-Learning update (off-policy bootstrap)
                best_next = np.max(self.Q[s_next]) if not done else 0.0
                td_target = r + self.gamma * best_next
                self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

                s       = s_next
                total_r += r

            ep_returns.append(total_r)

            # Periodic evaluation
            if (ep + 1) % eval_every == 0:
                result = evaluate_policy(self.get_policy(),
                                         n_episodes=eval_eps,
                                         seed=self.seed)
                eval_steps.append(ep + 1)
                eval_returns.append(result['mean_return'])
                eval_survival.append(result['survival_rate'])
                if verbose:
                    print(f"  QL ep {ep+1:6d} | ε={eps:.3f} | "
                          f"greedy return={result['mean_return']:.4f} | "
                          f"survival={result['survival_rate']*100:.1f}%")

        env.close()
        return {
            'episode_returns': np.array(ep_returns),
            'eval_steps':      np.array(eval_steps),
            'eval_returns':    np.array(eval_returns),
            'eval_survival':   np.array(eval_survival),
            'epsilons':        np.array(epsilons),
        }

    def get_policy(self) -> np.ndarray:
        """Greedy policy from current Q: (N_STATES,) int array."""
        return np.argmax(self.Q, axis=1)

    def get_Q(self) -> np.ndarray:
        """Return Q-table: (N_STATES, N_ACTIONS) float array."""
        return self.Q.copy()


# ─────────────────────────────────────────────────────────────
#  3. SARSA  (model-free, on-policy)
# ─────────────────────────────────────────────────────────────

class SARSA:
    """
    SARSA (State-Action-Reward-State-Action) — on-policy TD control.

    Updates toward the bootstrapped return of the *same* ε-greedy policy:
        Q(s, a) ← Q(s, a) + α [r + γ * Q(s', a') − Q(s, a)]
    where a' is selected by the current policy (not greedily).

    The on-policy nature makes SARSA more conservative than Q-Learning
    in stochastic environments because the bootstrap target accounts for
    exploratory actions. This is clinically relevant: a conservative agent
    that avoids extreme doses unless it is confident is safer in practice.

    Parameters
    ----------
    Same as QLearning.
    """

    def __init__(self,
                 n_states: int  = N_STATES,
                 n_actions: int = N_ACTIONS,
                 alpha: float   = 0.1,
                 gamma: float   = GAMMA,
                 epsilon_start: float        = 1.0,
                 epsilon_end:   float        = 0.05,
                 epsilon_decay_steps: int    = 5_000,
                 seed: int = 42):

        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = epsilon_start
        self.eps_end   = epsilon_end
        self.eps_decay = epsilon_decay_steps
        self.seed      = seed

        np.random.seed(seed)
        self.Q = np.zeros((n_states, n_actions))

    # ----------------------------------------------------------
    def _epsilon(self, episode: int) -> float:
        frac = min(episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _select_action(self, state: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    # ----------------------------------------------------------
    def train(self,
              n_episodes: int = 10_000,
              eval_every:  int = 500,
              eval_eps:    int = 300,
              verbose:     bool = True) -> dict:
        """
        Train the SARSA agent.

        Returns same structure as QLearning.train().
        """
        env = make_sepsis_env()

        ep_returns  = []
        eval_steps, eval_returns, eval_survival = [], [], []
        epsilons    = []

        for ep in range(n_episodes):
            eps = self._epsilon(ep)
            epsilons.append(eps)

            obs, _ = env.reset(seed=np.random.randint(100_000))
            s = int(obs)
            a = self._select_action(s, eps)          # SARSA selects a at reset
            total_r, done = 0.0, False

            while not done:
                obs_next, r, te, tr, _ = env.step(a)
                s_next = int(obs_next)
                done   = te or tr

                # Select NEXT action ON-POLICY (ε-greedy)
                a_next = self._select_action(s_next, eps) if not done else 0

                # SARSA update (on-policy bootstrap)
                td_target = r + self.gamma * (self.Q[s_next, a_next] if not done else 0.0)
                self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

                s       = s_next
                a       = a_next
                total_r += r

            ep_returns.append(total_r)

            # Periodic evaluation
            if (ep + 1) % eval_every == 0:
                result = evaluate_policy(self.get_policy(),
                                         n_episodes=eval_eps,
                                         seed=self.seed)
                eval_steps.append(ep + 1)
                eval_returns.append(result['mean_return'])
                eval_survival.append(result['survival_rate'])
                if verbose:
                    print(f"  SA ep {ep+1:6d} | ε={eps:.3f} | "
                          f"greedy return={result['mean_return']:.4f} | "
                          f"survival={result['survival_rate']*100:.1f}%")

        env.close()
        return {
            'episode_returns': np.array(ep_returns),
            'eval_steps':      np.array(eval_steps),
            'eval_returns':    np.array(eval_returns),
            'eval_survival':   np.array(eval_survival),
            'epsilons':        np.array(epsilons),
        }

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def get_Q(self) -> np.ndarray:
        return self.Q.copy()
