# %% [markdown]
# # Reinforcement Learning: Text Flappy Bird
# **3MD3220 – Individual Assignment**
# 
# **Author:** Adam Y.
# 
# This notebook implements and compares two reinforcement learning agents on the
# **TextFlappyBird-v0** environment:
# 1. **Monte Carlo (Every-Visit, ε-greedy)** — a model-free, episode-based method.
# 2. **Sarsa(λ)** — an on-line TD method with eligibility traces (Sutton & Barto, §12.7).
#
# Environment: `TextFlappyBird-v0` returns `(x_dist, y_dist)` — the horizontal and
# vertical distances from the player to the centre of the closest pipe gap.

# %% [markdown]
# ## 1. Setup & Environment Overview

# %%
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import text_flappy_bird_gym
from collections import defaultdict
import time
import os
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)

# %%
# --- Quick test of the environment ---
env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
obs, info = env.reset()
print(f"Observation space : {env.observation_space}")
print(f"Action space      : {env.action_space}   (0=idle, 1=flap)")
print(f"Initial obs (x,y) : {obs}")
print(f"x range: 0 .. {env.observation_space[0].n - 1}")
y_start = env.observation_space[1].start
y_end   = y_start + env.observation_space[1].n - 1
print(f"y range: {y_start} .. {y_end}")
env.close()

# %% [markdown]
# **Observation**: `(x_dist, y_dist)` where
# - `x_dist ∈ {0, …, 13}` — horizontal distance to the next pipe
# - `y_dist ∈ {-11, …, 10}` — vertical offset to the pipe-gap centre (negative = below)
#
# **Actions**: `0` = stay / idle, `1` = flap (jump up)
#
# **Reward**: `+1` for every time-step the bird is alive.

# %% [markdown]
# ## 2. Helper Utilities

# %%
def run_episode(env, policy_fn, max_steps=5000):
    """
    Run one episode using a given policy function.
    Returns the list of (state, action, reward) tuples and the total score.
    """
    obs, info = env.reset()
    state = tuple(obs) if not isinstance(obs, tuple) else obs
    trajectory = []
    total_reward = 0
    for _ in range(max_steps):
        action = policy_fn(state)
        next_obs, reward, done, _, info = env.step(action)
        trajectory.append((state, action, reward))
        total_reward += reward
        if done:
            break
        state = tuple(next_obs) if not isinstance(next_obs, tuple) else next_obs
    return trajectory, total_reward, info.get("score", 0)


def evaluate_agent(env, policy_fn, n_episodes=50):
    """Evaluate a greedy policy over n_episodes. Returns mean and std of rewards."""
    rewards = []
    scores = []
    for _ in range(n_episodes):
        _, r, s = run_episode(env, policy_fn)
        rewards.append(r)
        scores.append(s)
    return np.mean(rewards), np.std(rewards), np.mean(scores), np.std(scores)


def moving_average(data, window=100):
    """Compute a smoothed moving average."""
    if len(data) < window:
        window = max(1, len(data) // 5)
    return np.convolve(data, np.ones(window)/window, mode='valid')

# %% [markdown]
# ## 3. Agent 1 — Monte Carlo (Every-Visit, ε-greedy)
#
# We use the **every-visit MC control** method with ε-greedy exploration.
# The algorithm updates Q(s,a) at the end of each episode using the returns
# observed from every visit to each (state, action) pair.

# %%
class MonteCarloAgent:
    """
    Every-Visit Monte Carlo Control with epsilon-greedy policy.
    Uses incremental mean updates for efficiency.
    """
    def __init__(self, n_actions=2, gamma=1.0, epsilon=0.1, epsilon_decay=0.9999, epsilon_min=0.01):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table and visit counts
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.N = defaultdict(lambda: np.zeros(n_actions))  # visit counts
    
    def epsilon_greedy_action(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))
    
    def greedy_action(self, state):
        """Select best action (greedy)."""
        return int(np.argmax(self.Q[state]))
    
    def train_episode(self, env, max_steps=5000):
        """
        Run one episode, collect trajectory, then update Q for every visit.
        Returns total reward and game score.
        """
        trajectory, total_reward, score = run_episode(
            env, self.epsilon_greedy_action, max_steps
        )
        
        # Compute returns and update Q values (every-visit)
        G = 0
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = self.gamma * G + reward
            self.N[state][action] += 1
            # Incremental mean update
            alpha = 1.0 / self.N[state][action]
            self.Q[state][action] += alpha * (G - self.Q[state][action])
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward, score
    
    def train(self, env, n_episodes=10000, log_every=500):
        """Train the agent for n_episodes. Returns reward history."""
        reward_history = []
        score_history = []
        for ep in range(1, n_episodes + 1):
            r, s = self.train_episode(env)
            reward_history.append(r)
            score_history.append(s)
            if ep % log_every == 0:
                avg_r = np.mean(reward_history[-log_every:])
                avg_s = np.mean(score_history[-log_every:])
                print(f"  Episode {ep:>6}/{n_episodes}  |  "
                      f"avg reward: {avg_r:>7.1f}  |  avg score: {avg_s:>5.1f}  |  "
                      f"eps: {self.epsilon:.4f}  |  Q-table size: {len(self.Q)}")
        return reward_history, score_history

# %% [markdown]
# ### 3.1 Training the Monte Carlo Agent

# %%
# Create environment
env_mc = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

# Create and train the agent
mc_agent = MonteCarloAgent(
    n_actions=2,
    gamma=1.0,
    epsilon=1.0,
    epsilon_decay=0.9998,
    epsilon_min=0.01
)

print("=== Training Monte Carlo Agent ===")
mc_rewards, mc_scores = mc_agent.train(env_mc, n_episodes=15000, log_every=1000)
env_mc.close()

# %% [markdown]
# ### 3.2 Evaluate Monte Carlo Agent

# %%
env_eval = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
mean_r, std_r, mean_s, std_s = evaluate_agent(env_eval, mc_agent.greedy_action, n_episodes=100)
print(f"MC Agent — Mean reward: {mean_r:.1f} ± {std_r:.1f}  |  Mean score: {mean_s:.1f} ± {std_s:.1f}")
env_eval.close()

# %% [markdown]
# ## 4. Agent 2 — Tabular Sarsa(λ)
#
# Implementation following Sutton & Barto, §12.7.
# Since the TextFlappyBird-v0 state space is small (14 × 22 = 308 states),
# we use a **tabular** representation — a Q-dictionary indexed by `(state, action)` —
# with **accumulating eligibility traces** for efficient credit assignment.
# This provides a fair apples-to-apples comparison with the tabular MC agent.

# %% [markdown]
# ### 4.1 Sarsa(λ) Agent

# %%
class SarsaLambdaAgent:
    """
    Tabular Sarsa(lambda) with accumulating eligibility traces.
    Follows Sutton & Barto §12.7 (tabular version).

    For each (state, action) pair we maintain:
      - Q[state][action] : action-value estimate
      - During each episode, an eligibility trace dict z[(s,a)]
    """
    def __init__(self, n_actions=2,
                 alpha=0.1, gamma=1.0, lam=0.9,
                 epsilon=0.1, epsilon_decay=0.9999, epsilon_min=0.01):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> array of shape (n_actions,)
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def epsilon_greedy_action(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state):
        """Select best action (greedy)."""
        return int(np.argmax(self.Q[state]))

    def train_episode(self, env, max_steps=5000):
        """
        Run one episode of tabular Sarsa(lambda) with accumulating traces.
        """
        obs, info = env.reset()
        state = tuple(obs) if not isinstance(obs, tuple) else obs
        action = self.epsilon_greedy_action(state)

        # Eligibility traces — sparse dict (only visited pairs stored)
        z = defaultdict(lambda: np.zeros(self.n_actions))

        total_reward = 0
        score = 0

        for step in range(max_steps):
            # Take action A, observe R, S'
            next_obs, reward, done, _, info = env.step(action)
            total_reward += reward
            score = info.get("score", 0)

            if done:
                # TD error at terminal: delta = R - Q(S, A)
                delta = reward - self.Q[state][action]
                # Update trace for current (S, A)
                z[state][action] += 1
                # Update all visited Q entries
                for s in list(z.keys()):
                    self.Q[s] += self.alpha * delta * z[s]
                break

            next_state = tuple(next_obs) if not isinstance(next_obs, tuple) else next_obs
            next_action = self.epsilon_greedy_action(next_state)

            # TD error: delta = R + gamma * Q(S', A') - Q(S, A)
            delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]

            # Accumulating trace for current (S, A)
            z[state][action] += 1

            # Update all visited Q entries and decay traces
            keys_to_delete = []
            for s in list(z.keys()):
                self.Q[s] += self.alpha * delta * z[s]
                z[s] *= self.gamma * self.lam
                # Prune near-zero traces for efficiency
                if np.max(np.abs(z[s])) < 1e-6:
                    keys_to_delete.append(s)
            for s in keys_to_delete:
                del z[s]

            state = next_state
            action = next_action

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return total_reward, score

    def train(self, env, n_episodes=10000, log_every=500):
        """Train the agent for n_episodes. Returns reward history."""
        reward_history = []
        score_history = []
        for ep in range(1, n_episodes + 1):
            r, s = self.train_episode(env)
            reward_history.append(r)
            score_history.append(s)
            if ep % log_every == 0:
                avg_r = np.mean(reward_history[-log_every:])
                avg_s = np.mean(score_history[-log_every:])
                print(f"  Episode {ep:>6}/{n_episodes}  |  "
                      f"avg reward: {avg_r:>7.1f}  |  avg score: {avg_s:>5.1f}  |  "
                      f"eps: {self.epsilon:.4f}  |  Q-table size: {len(self.Q)}")
        return reward_history, score_history

# %% [markdown]
# ### 4.2 Training the Sarsa(λ) Agent

# %%
# Create environment
env_sarsa = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

# Create and train the agent
sarsa_agent = SarsaLambdaAgent(
    n_actions=2,
    alpha=0.1,       # learning rate
    gamma=1.0,
    lam=0.9,         # lambda for eligibility traces
    epsilon=1.0,
    epsilon_decay=0.9998,
    epsilon_min=0.01
)

print("=== Training Sarsa(lambda) Agent ===")
sarsa_rewards, sarsa_scores = sarsa_agent.train(env_sarsa, n_episodes=15000, log_every=1000)
env_sarsa.close()

# %% [markdown]
# ### 4.3 Evaluate Sarsa(λ) Agent

# %%
env_eval2 = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
mean_r, std_r, mean_s, std_s = evaluate_agent(env_eval2, sarsa_agent.greedy_action, n_episodes=100)
print(f"Sarsa(λ) Agent — Mean reward: {mean_r:.1f} ± {std_r:.1f}  |  Mean score: {mean_s:.1f} ± {std_s:.1f}")
env_eval2.close()

# %% [markdown]
# ## 5. Results Comparison

# %% [markdown]
# ### 5.1 Learning Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Reward over episodes ---
ax = axes[0]
window = 200
mc_smooth = moving_average(mc_rewards, window)
sarsa_smooth = moving_average(sarsa_rewards, window)
ax.plot(mc_smooth, label='Monte Carlo', alpha=0.9, linewidth=1.5)
ax.plot(sarsa_smooth, label='Sarsa(λ)', alpha=0.9, linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward (smoothed)')
ax.set_title('Learning Curve — Reward per Episode')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Score over episodes ---
ax = axes[1]
mc_score_smooth = moving_average(mc_scores, window)
sarsa_score_smooth = moving_average(sarsa_scores, window)
ax.plot(mc_score_smooth, label='Monte Carlo', alpha=0.9, linewidth=1.5)
ax.plot(sarsa_score_smooth, label='Sarsa(λ)', alpha=0.9, linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Game Score (smoothed)')
ax.set_title('Learning Curve — Score per Episode')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: learning_curves.png")

# %% [markdown]
# ### 5.2 State-Value Function Plots
#
# We plot $V(s) = \max_a Q(s, a)$ for each state in the observation grid.

# %%
# --- MC Value Function ---
x_range = range(0, 14)
y_range = range(-11, 11)

V_mc = np.zeros((len(y_range), len(x_range)))
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        state = (x, y)
        V_mc[i, j] = np.max(mc_agent.Q[state])

# --- Sarsa(λ) Value Function ---
V_sarsa = np.zeros((len(y_range), len(x_range)))
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        state = (x, y)
        V_sarsa[i, j] = np.max(sarsa_agent.Q[state])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MC value function
ax = axes[0]
im = ax.imshow(V_mc, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, 13, -11, 10])
ax.set_xlabel('x distance to pipe')
ax.set_ylabel('y distance to pipe gap')
ax.set_title('State-Value Function V(s) — Monte Carlo')
plt.colorbar(im, ax=ax, label='V(s)')

# Sarsa(λ) value function
ax = axes[1]
im = ax.imshow(V_sarsa, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, 13, -11, 10])
ax.set_xlabel('x distance to pipe')
ax.set_ylabel('y distance to pipe gap')
ax.set_title('State-Value Function V(s) — Sarsa(λ)')
plt.colorbar(im, ax=ax, label='V(s)')

plt.tight_layout()
plt.savefig('value_functions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: value_functions.png")

# %% [markdown]
# ### 5.3 Action-Value Function & Learned Policy

# %%
# --- Best action map for both agents ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MC policy
policy_mc = np.zeros((len(y_range), len(x_range)))
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        state = (x, y)
        policy_mc[i, j] = mc_agent.greedy_action(state)

ax = axes[0]
im = ax.imshow(policy_mc, aspect='auto', origin='lower', cmap='coolwarm',
               extent=[0, 13, -11, 10], vmin=0, vmax=1)
ax.set_xlabel('x distance to pipe')
ax.set_ylabel('y distance to pipe gap')
ax.set_title('Learned Policy — Monte Carlo\n(blue=idle, red=flap)')
plt.colorbar(im, ax=ax, ticks=[0, 1], label='Action')

# Sarsa policy
policy_sarsa = np.zeros((len(y_range), len(x_range)))
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        state = (x, y)
        policy_sarsa[i, j] = sarsa_agent.greedy_action(state)

ax = axes[1]
im = ax.imshow(policy_sarsa, aspect='auto', origin='lower', cmap='coolwarm',
               extent=[0, 13, -11, 10], vmin=0, vmax=1)
ax.set_xlabel('x distance to pipe')
ax.set_ylabel('y distance to pipe gap')
ax.set_title('Learned Policy — Sarsa(λ)\n(blue=idle, red=flap)')
plt.colorbar(im, ax=ax, ticks=[0, 1], label='Action')

plt.tight_layout()
plt.savefig('learned_policies.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: learned_policies.png")

# %% [markdown]
# ### 5.4 Q-Value Difference between Actions
#
# Plotting $Q(s, 1) - Q(s, 0)$ (positive = flap preferred, negative = idle preferred).

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MC Q difference
Q_diff_mc = np.zeros((len(y_range), len(x_range)))
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        state = (x, y)
        Q_diff_mc[i, j] = mc_agent.Q[state][1] - mc_agent.Q[state][0]

ax = axes[0]
vmax = max(abs(Q_diff_mc.min()), abs(Q_diff_mc.max())) or 1
im = ax.imshow(Q_diff_mc, aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[0, 13, -11, 10], vmin=-vmax, vmax=vmax)
ax.set_xlabel('x distance to pipe')
ax.set_ylabel('y distance to pipe gap')
ax.set_title('Q(s,flap) − Q(s,idle) — Monte Carlo')
plt.colorbar(im, ax=ax, label='ΔQ')

# Sarsa Q difference
Q_diff_sarsa = np.zeros((len(y_range), len(x_range)))
for i, y in enumerate(y_range):
    for j, x in enumerate(x_range):
        state = (x, y)
        Q_diff_sarsa[i, j] = sarsa_agent.Q[state][1] - sarsa_agent.Q[state][0]

ax = axes[1]
vmax = max(abs(Q_diff_sarsa.min()), abs(Q_diff_sarsa.max())) or 1
im = ax.imshow(Q_diff_sarsa, aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[0, 13, -11, 10], vmin=-vmax, vmax=vmax)
ax.set_xlabel('x distance to pipe')
ax.set_ylabel('y distance to pipe gap')
ax.set_title('Q(s,flap) − Q(s,idle) — Sarsa(λ)')
plt.colorbar(im, ax=ax, label='ΔQ')

plt.tight_layout()
plt.savefig('q_difference.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: q_difference.png")

# %% [markdown]
# ## 6. Parameter Sweeps

# %% [markdown]
# ### 6.1 MC — Epsilon Decay Rate Sweep

# %%
decay_rates = [0.9990, 0.9995, 0.9998, 0.99995]
mc_sweep_results = {}

for decay in decay_rates:
    print(f"\n--- MC sweep: epsilon_decay={decay} ---")
    env_sw = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    agent = MonteCarloAgent(n_actions=2, gamma=1.0, epsilon=1.0,
                            epsilon_decay=decay, epsilon_min=0.01)
    rewards, scores = agent.train(env_sw, n_episodes=10000, log_every=5000)
    mc_sweep_results[decay] = (rewards, scores)
    env_sw.close()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
for decay, (rewards, _) in mc_sweep_results.items():
    smoothed = moving_average(rewards, 200)
    ax.plot(smoothed, label=f'ε-decay={decay}', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward (smoothed)')
ax.set_title('MC Agent — Epsilon Decay Rate Sweep')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mc_epsilon_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.2 Sarsa(λ) — Lambda Sweep

# %%
lambda_values = [0.0, 0.3, 0.6, 0.9, 0.95]
sarsa_sweep_results = {}

for lam in lambda_values:
    print(f"\n--- Sarsa sweep: lambda={lam} ---")
    env_sw = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    agent = SarsaLambdaAgent(n_actions=2, alpha=0.1,
                             gamma=1.0, lam=lam, epsilon=1.0,
                             epsilon_decay=0.9998, epsilon_min=0.01)
    rewards, scores = agent.train(env_sw, n_episodes=10000, log_every=5000)
    sarsa_sweep_results[lam] = (rewards, scores)
    env_sw.close()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
for lam, (rewards, _) in sarsa_sweep_results.items():
    smoothed = moving_average(rewards, 200)
    ax.plot(smoothed, label=f'λ={lam}', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward (smoothed)')
ax.set_title('Sarsa(λ) Agent — Lambda Sweep')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sarsa_lambda_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.3 Sarsa(λ) — Learning Rate Sweep

# %%
alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
sarsa_alpha_results = {}

for alpha in alpha_values:
    print(f"\n--- Sarsa sweep: alpha={alpha} ---")
    env_sw = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    agent = SarsaLambdaAgent(n_actions=2, alpha=alpha,
                             gamma=1.0, lam=0.9, epsilon=1.0,
                             epsilon_decay=0.9998, epsilon_min=0.01)
    rewards, scores = agent.train(env_sw, n_episodes=10000, log_every=5000)
    sarsa_alpha_results[alpha] = (rewards, scores)
    env_sw.close()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
for alpha, (rewards, _) in sarsa_alpha_results.items():
    smoothed = moving_average(rewards, 200)
    ax.plot(smoothed, label=f'α={alpha}', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward (smoothed)')
ax.set_title('Sarsa(λ) Agent — Learning Rate Sweep')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sarsa_alpha_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Generalization: Trained Agent on Different Level Configurations
#
# We test how well agents trained on `(height=15, width=20, pipe_gap=4)` perform
# on different environment configurations.

# %%
configs = [
    {"height": 15, "width": 20, "pipe_gap": 4},   # original (training config)
    {"height": 15, "width": 20, "pipe_gap": 3},   # smaller gap
    {"height": 15, "width": 20, "pipe_gap": 5},   # larger gap
    {"height": 10, "width": 20, "pipe_gap": 4},   # shorter screen
    {"height": 20, "width": 20, "pipe_gap": 4},   # taller screen
    {"height": 15, "width": 30, "pipe_gap": 4},   # wider screen
]

print("=== Generalization test ===")
print(f"{'Config':<35} | {'MC reward':>12} | {'MC score':>10} | {'Sarsa reward':>14} | {'Sarsa score':>12}")
print("-" * 100)

gen_results = []
for cfg in configs:
    env_gen = gym.make('TextFlappyBird-v0', **cfg)
    
    mc_mr, mc_sr, mc_ms, mc_ss = evaluate_agent(env_gen, mc_agent.greedy_action, n_episodes=50)
    sa_mr, sa_sr, sa_ms, sa_ss = evaluate_agent(env_gen, sarsa_agent.greedy_action, n_episodes=50)
    
    cfg_str = f"h={cfg['height']}, w={cfg['width']}, gap={cfg['pipe_gap']}"
    print(f"{cfg_str:<35} | {mc_mr:>7.1f}±{mc_sr:>4.1f} | {mc_ms:>5.1f}±{mc_ss:>3.1f} | "
          f"{sa_mr:>8.1f}±{sa_sr:>5.1f} | {sa_ms:>5.1f}±{sa_ss:>4.1f}")
    gen_results.append((cfg_str, mc_mr, mc_ms, sa_mr, sa_ms))
    env_gen.close()

# Bar plot of generalization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = [r[0] for r in gen_results]
mc_r = [r[1] for r in gen_results]
sa_r = [r[3] for r in gen_results]
mc_s = [r[2] for r in gen_results]
sa_s = [r[4] for r in gen_results]

x = np.arange(len(labels))
width = 0.35

ax = axes[0]
ax.bar(x - width/2, mc_r, width, label='Monte Carlo', alpha=0.8)
ax.bar(x + width/2, sa_r, width, label='Sarsa(λ)', alpha=0.8)
ax.set_xlabel('Configuration')
ax.set_ylabel('Mean Reward')
ax.set_title('Generalization — Reward')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(x - width/2, mc_s, width, label='Monte Carlo', alpha=0.8)
ax.bar(x + width/2, sa_s, width, label='Sarsa(λ)', alpha=0.8)
ax.set_xlabel('Configuration')
ax.set_ylabel('Mean Score')
ax.set_title('Generalization — Score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('generalization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: generalization.png")

# %% [markdown]
# ## 8. Comparison of TextFlappyBird-v0 vs TextFlappyBird-screen-v0
#
# We briefly test the screen-based environment to compare with the distance-based one.

# %%
# Check the screen environment observation space
env_screen = gym.make('TextFlappyBird-screen-v0', height=15, width=20, pipe_gap=4)
obs_screen, info = env_screen.reset()
print(f"Screen env observation space: {env_screen.observation_space}")
print(f"Screen obs shape: {np.array(obs_screen).shape}")
print(f"Screen obs dtype: {np.array(obs_screen).dtype}")

# Show a sample observation
obs_arr = np.array(obs_screen)
print(f"\nSample screen observation (first 5 rows):")
print(obs_arr[:5])
env_screen.close()

# %% [markdown]
# The **screen** environment returns a large 2D array (15×20 in this config) where
# each cell encodes the game character at that position. This makes tabular methods
# infeasible due to the enormous state space ($\sim 4^{300}$ possible states).
# The **v0** environment, which simply returns `(x_dist, y_dist)`, has at most
# $14 \times 22 = 308$ possible states, making it ideal for tabular MC and
# tabular Sarsa(λ).

# %% [markdown]
# ## 9. Discussion: Applicability to Original Flappy Bird
#
# The original Flappy Bird game (via `flappy-bird-gymnasium`) uses *continuous* pixel
# observations (RGB) or continuous position values with floating-point precision.
# 
# **Can the same agents be used?**
# - The **tabular MC agent** cannot be directly used because the continuous state
#   space makes it impossible to maintain a Q-table.
# - The **Sarsa(λ) agent** with tile coding can be adapted, as tile coding discretises
#   continuous states — though the observation features may need to be extracted
#   differently (e.g., pipe positions, velocity).
# - For the pixel-based original game, **deep RL** methods (DQN, PPO) would be
#   needed to handle the high-dimensional observation space.

# %% [markdown]
# ## 10. Summary Table

# %%
# Final evaluation with more episodes
env_final = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

mc_mr, mc_sr, mc_ms, mc_ss = evaluate_agent(env_final, mc_agent.greedy_action, n_episodes=200)
sa_mr, sa_sr, sa_ms, sa_ss = evaluate_agent(env_final, sarsa_agent.greedy_action, n_episodes=200)

print("=" * 65)
print(f"{'Agent':<20} | {'Mean Reward':>12} | {'Std Reward':>10} | {'Mean Score':>10} | {'Std Score':>10}")
print("-" * 65)
print(f"{'Monte Carlo':<20} | {mc_mr:>12.1f} | {mc_sr:>10.1f} | {mc_ms:>10.1f} | {mc_ss:>10.1f}")
print(f"{'Sarsa(λ)':<20} | {sa_mr:>12.1f} | {sa_sr:>10.1f} | {sa_ms:>10.1f} | {sa_ss:>10.1f}")
print("=" * 65)

env_final.close()

# %% [markdown]
# ## 11. Conclusions
#
# 1. **Both agents successfully learn** to play TextFlappyBird-v0, demonstrating
#    that RL can solve this simple game.
#
# 2. **Monte Carlo** is conceptually simpler (tabular Q-values updated per episode)
#    but requires complete episodes and has higher variance in early training.
#
# 3. **Sarsa(λ)** with eligibility traces provides faster credit assignment
#    (TD updates within the episode). Both agents use tabular representations,
#    which is well-suited to the small discrete state space (308 states).
#
# 4. **Parameter sensitivity**: MC is mainly sensitive to epsilon decay; Sarsa(λ)
#    is sensitive to λ and α — large values of both can cause instability.
#
# 5. **Generalization**: Both agents struggle when the environment configuration
#    changes significantly (e.g., smaller pipe gap), since both are tabular and
#    only have knowledge of states seen during training.
#
# 6. **Screen vs distance environment**: The distance-based `v0` environment is
#    much more suitable for these tabular methods, while the screen version
#    would require deep RL approaches.

print("\n=== Notebook complete ===")
