"""Custom PPO + GAE training loop for Isaac-Cartpole-v0."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

# ---------------- 超參數 ----------------
device = "cuda:0"
num_steps = 32
gamma = 0.99
lam = 0.95
clip_eps = 0.2
lr = 3e-4
epochs_per_update = 5
minibatch_size = 2048
ent_coef = 0.01
val_coef = 0.5

# ---------------- 建立環境 ----------------
cfg = load_cfg_from_registry("Isaac-Cartpole-v0", "env_cfg_entry_point")
cfg.scene.num_envs = args_cli.num_envs
env = gym.make("Isaac-Cartpole-v0", cfg=cfg)

obs_dim = env.observation_space["policy"].shape[-1]
act_dim = env.action_space.shape[-1]
num_envs = args_cli.num_envs

print(f"[INFO]: obs_dim={obs_dim}, act_dim={act_dim}, num_envs={num_envs}")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_action_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, logprob, entropy, value


model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

obs_dict, _ = env.reset()
obs = obs_dict["policy"]

for iteration in range(args_cli.iterations):
    b_obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    b_actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    b_logprobs = torch.zeros((num_steps, num_envs), device=device)
    b_rewards = torch.zeros((num_steps, num_envs), device=device)
    b_dones = torch.zeros((num_steps, num_envs), device=device)
    b_values = torch.zeros((num_steps, num_envs), device=device)

    for t in range(num_steps):
        with torch.no_grad():
            action, logprob, _, value = model.get_action_value(obs)
        next_obs_dict, reward, terminated, truncated, _ = env.step(action)
        done = (terminated | truncated).float()

        b_obs[t] = obs
        b_actions[t] = action
        b_logprobs[t] = logprob
        b_values[t] = value
        b_rewards[t] = reward
        b_dones[t] = done

        obs = next_obs_dict["policy"]

    with torch.no_grad():
        _, _, _, last_value = model.get_action_value(obs)
        advantages = torch.zeros_like(b_rewards)
        lastgaelam = torch.zeros(num_envs, device=device)
        for t in reversed(range(num_steps)):
            nextnonterminal = 1.0 - (b_dones[t])
            nextvalue = last_value if t == num_steps - 1 else b_values[t + 1]
            delta = b_rewards[t] + gamma * nextvalue * nextnonterminal - b_values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + b_values

    flat_obs = b_obs.reshape(-1, obs_dim)
    flat_actions = b_actions.reshape(-1, act_dim)
    flat_logprobs = b_logprobs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)

    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

    total_samples = flat_obs.shape[0]
    indices = torch.randperm(total_samples, device=device)

    for epoch in range(epochs_per_update):
        for start in range(0, total_samples, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            _, new_logprob, entropy, new_value = model.get_action_value(
                flat_obs[mb_idx], flat_actions[mb_idx]
            )
            ratio = (new_logprob - flat_logprobs[mb_idx]).exp()

            mb_adv = flat_advantages[mb_idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((new_value - flat_returns[mb_idx]) ** 2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + val_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    mean_reward = b_rewards.mean().item()
    print(f"[Iter {iteration}] mean_reward={mean_reward:.3f} policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}")

env.close()
simulation_app.close()
