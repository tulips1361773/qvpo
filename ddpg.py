# ddpg_v2.py
import argparse
import os
import random
import time
import datetime
from collections import deque
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 导入自定义环境
from myenv import UAVISACEnvironment

def parse_args():
    parser = argparse.ArgumentParser(description='DDPG for UAV-ISAC Improved')
    
    # --- 环境特定参数 ---
    parser.add_argument('--env_name', default="Env", help='Environment ID')
    # ... (保持原有的环境参数不变，方便复用) ...
    parser.add_argument('--eav_agg', type=str, default='logsumexp', choices=['max', 'top2', 'logsumexp'])
    parser.add_argument('--eav_logsumexp_kappa', type=float, default=0.5)
    parser.add_argument('--eav_threshold', type=float, default=10.0)
    parser.add_argument('--eav_penalty_coef', type=float, default=3.0)
    parser.add_argument('--eav_penalty_cap', type=float, default=20.0)
    parser.add_argument('--comm_penalty', type=str, default='softplus', choices=['hinge', 'softplus', 'huber'])
    parser.add_argument('--comm_threshold', type=float, default=10.0)
    parser.add_argument('--comm_penalty_coef', type=float, default=1.5)
    parser.add_argument('--comm_softplus_kappa', type=float, default=5.0)
    parser.add_argument('--comm_huber_delta', type=float, default=1.0)
    parser.add_argument('--comm_penalty_cap_per_user', type=float, default=15.0)
    parser.add_argument('--comm_penalty_cap_total', type=float, default=30.0)
    parser.add_argument('--comm_penalty_avg_over_k', type=lambda x: bool(strtobool(str(x))), default=True)
    parser.add_argument('--action_smooth_coef', type=float, default=0.8)
    parser.add_argument('--user_move_range', type=float, default=20.0)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    parser.add_argument('--eta_clip_max', type=float, default=15.0)
    parser.add_argument('--comm_penalty_clip_max', type=float, default=5.0)
    parser.add_argument('--eav_penalty_clip_max', type=float, default=5.0)
    
    # ⚠️ 强制关闭外部状态归一化，改用 LayerNorm
    parser.add_argument('--normalize_state', type=lambda x: bool(strtobool(str(x))), default=False)

    # --- DDPG 算法参数 (优化版) ---
    parser.add_argument('--exp_name', type=str, default="ddpg_v2")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--total_timesteps', type=int, default=1000000) # 减少一点步数，因为收敛应该变快
    parser.add_argument('--actor_lr', type=float, default=1e-4)  # 降低学习率
    parser.add_argument('--critic_lr', type=float, default=1e-3) # Critic 学习率稍大
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--learning_starts', type=int, default=10000)
    parser.add_argument('--policy_frequency', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-2) # 🔥 新增：权重衰减
    
    args = parser.parse_args()
    return args

# --- 改进的网络定义 (加入 LayerNorm) ---
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        
        # 🔥 LayerNorm 直接处理输入，代替 StateNormalizer
        self.layer_norm = nn.LayerNorm(obs_dim)
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        
        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.layer_norm(x) # 归一化输入
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        
        # 🔥 LayerNorm 处理状态输入
        self.layer_norm = nn.LayerNorm(obs_dim)
        
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.layer_norm(x) # 归一化输入
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def make_env(args):
    # 强制 normalize_state=False，防止 Buffer 污染
    env = UAVISACEnvironment(
        normalize_state=False, # 🔥 关键修改
        eav_agg=args.eav_agg,
        eav_logsumexp_kappa=args.eav_logsumexp_kappa,
        eav_threshold=args.eav_threshold,
        eav_penalty_coef=args.eav_penalty_coef,
        eav_penalty_cap=args.eav_penalty_cap,
        comm_penalty_type=args.comm_penalty,
        comm_threshold=args.comm_threshold,
        comm_penalty_coef=args.comm_penalty_coef,
        comm_softplus_kappa=args.comm_softplus_kappa,
        comm_huber_delta=args.comm_huber_delta,
        comm_penalty_cap_per_user=args.comm_penalty_cap_per_user,
        comm_penalty_cap_total=args.comm_penalty_cap_total,
        comm_penalty_avg_over_k=args.comm_penalty_avg_over_k,
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
        reward_scale=args.reward_scale,
        eta_clip_max=args.eta_clip_max,
        comm_penalty_clip_max=args.comm_penalty_clip_max,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
    )
    return env

def evaluate(actor, eval_env, device, steps):
    episodes = 5
    returns = []
    for _ in range(episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            next_state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
        returns.append(episode_reward)
    return np.mean(returns), np.std(returns)

def main():
    args = parse_args()
    
    run_name = f"{args.env_name}_DDPG_Improved_{args.seed}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join("record", args.env_name, "DDPG_Imp", f"seed={args.seed}")
    writer = SummaryWriter(log_dir)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    # 创建环境 (不再区分 train/eval 的归一化，因为现在是 Raw State)
    env = make_env(args)
    eval_env = make_env(args)

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())

    # 🔥 使用 Weight Decay 防止过拟合
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.critic_lr, weight_decay=args.weight_decay)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_lr) # Actor 通常不需要 WD，或者给很小

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    rb_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_actions = np.zeros((args.buffer_size, *act_shape), dtype=np.float32)
    rb_rewards = np.zeros((args.buffer_size), dtype=np.float32)
    rb_next_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_dones = np.zeros((args.buffer_size), dtype=np.float32)
    
    rb_pos = 0
    rb_size = 0

    obs, _ = env.reset(seed=args.seed)
    current_episode_reward = 0
    recent_rewards = deque(maxlen=100)
    ema_reward = None

    print(f"Starting improved DDPG training on {device}...")

    for global_step in range(args.total_timesteps):
        
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = actor(obs_tensor)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                noise = np.random.normal(0, env.action_space.high * args.exploration_noise, size=act_shape)
                action = (action + noise).clip(env.action_space.low, env.action_space.high)
                action = action.flatten()

        next_obs, reward, done, truncated, info = env.step(action)
        current_episode_reward += reward

        if global_step % 200 == 0:
            writer.add_scalar('reward_terms/reward_final', float(reward), global_step)
            # 记录其他详细指标... (省略以保持代码简洁，可复制之前的)
            writer.add_scalar('reward_terms/eta_0', float(info.get('eta_0', 0.0)), global_step)

        rb_obs[rb_pos] = obs
        rb_actions[rb_pos] = action
        rb_rewards[rb_pos] = reward
        rb_next_obs[rb_pos] = next_obs
        rb_dones[rb_pos] = done and not truncated
        
        rb_pos = (rb_pos + 1) % args.buffer_size
        rb_size = min(rb_size + 1, args.buffer_size)

        obs = next_obs

        if done or truncated:
            recent_rewards.append(current_episode_reward)
            ema_reward = current_episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * current_episode_reward)
            writer.add_scalar('reward/train', current_episode_reward, global_step)
            writer.add_scalar('reward/train_ema', ema_reward, global_step)
            obs, _ = env.reset(seed=args.seed + global_step)
            current_episode_reward = 0

        if global_step > args.learning_starts:
            batch_inds = np.random.randint(0, rb_size, size=args.batch_size)
            
            b_obs = torch.tensor(rb_obs[batch_inds], dtype=torch.float32).to(device)
            b_actions = torch.tensor(rb_actions[batch_inds], dtype=torch.float32).to(device)
            b_rewards = torch.tensor(rb_rewards[batch_inds], dtype=torch.float32).to(device).unsqueeze(1)
            b_next_obs = torch.tensor(rb_next_obs[batch_inds], dtype=torch.float32).to(device)
            b_dones = torch.tensor(rb_dones[batch_inds], dtype=torch.float32).to(device).unsqueeze(1)

            with torch.no_grad():
                next_state_actions = target_actor(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs, next_state_actions)
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * qf1_next_target
            
            qf1_a_values = qf1(b_obs, b_actions)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            q_optimizer.zero_grad()
            qf1_loss.backward()
            # 🔥 梯度裁剪：防止 Q Loss 爆炸导致网络权重崩坏
            torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=10.0)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(b_obs, actor(b_obs)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                actor_optimizer.step()

                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        if global_step > 0 and global_step % 10000 == 0:
            mean_ret, std_ret = evaluate(actor, eval_env, device, global_step)
            writer.add_scalar('reward/eval_mean', mean_ret, global_step)
            print(f"Step {global_step}: Eval Reward {mean_ret:.2f} +/- {std_ret:.2f}")

    env.close()
    eval_env.close()
    writer.close()

if __name__ == "__main__":
    main()