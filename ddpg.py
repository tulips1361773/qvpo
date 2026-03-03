# ddpg.py
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
    parser = argparse.ArgumentParser(description='DDPG for UAV-ISAC')
    
    # --- 环境特定参数 (与 main.py 保持一致) ---
    parser.add_argument('--env_name', default="Env", help='Environment ID')
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
    
    # 归一化参数
    parser.add_argument('--normalize_state', type=lambda x: bool(strtobool(str(x))), default=True)

    # --- DDPG 算法参数 (CleanRL 风格) ---
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=str, default="cuda:0")  # 兼容 main.py 的参数名
    parser.add_argument('--total_timesteps', type=int, default=2500000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--learning_starts', type=int, default=10000) # 对应 main.py 的 start_steps
    parser.add_argument('--policy_frequency', type=int, default=2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    
    # 评估与日志
    parser.add_argument('--eval_freq', type=int, default=10000)
    
    args = parser.parse_args()
    return args

# --- 网络定义 ---
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # 获取维度
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        
        # 动作范围缩放 (Env 已经是 -1 到 1，但为了保险读取 low/high)
        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 工具函数 ---
def make_env(args, is_eval=False):
    """
    根据参数创建 UAVISACEnvironment
    """
    env = UAVISACEnvironment(
        normalize_state=args.normalize_state,
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
    
    # 如果是评估环境，关闭归一化器的更新，防止评估数据污染统计量
    if is_eval and hasattr(env, 'state_normalizer'):
        env.state_normalizer.set_training(False)
        
    return env

def sync_normalization_stats(source_env, target_env):
    """
    将训练环境的归一化统计量同步到评估环境
    这是 DDPG 复用内部归一化的关键步骤！
    """
    if hasattr(source_env, 'state_normalizer') and hasattr(target_env, 'state_normalizer'):
        target_env.state_normalizer.mean = source_env.state_normalizer.mean.copy()
        target_env.state_normalizer.var = source_env.state_normalizer.var.copy()
        target_env.state_normalizer.count = source_env.state_normalizer.count
        # 确保评估环境处于非训练模式
        target_env.state_normalizer.set_training(False)

def evaluate(agent, eval_env, device, steps):
    episodes = 10
    returns = []
    
    for _ in range(episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                # DDPG 评估时不加噪声
                action = agent(state_tensor).cpu().numpy().flatten()
            
            next_state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
            
        returns.append(episode_reward)
    
    return np.mean(returns), np.std(returns)

def main():
    args = parse_args()
    
    # 目录设置
    run_name = f"DDPG_{args.env_name}_seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 保持和 main.py 类似的目录结构以便对比
    log_dir = os.path.join("record", args.env_name, "DDPG", f"seed={args.seed}")
    writer = SummaryWriter(log_dir)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    # 创建环境
    print("Initializing Environments...")
    env = make_env(args, is_eval=False)
    eval_env = make_env(args, is_eval=True)

    # 模型初始化
    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # 经验回放池 (CleanRL style implementation using numpy)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    rb_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_actions = np.zeros((args.buffer_size, *act_shape), dtype=np.float32)
    rb_rewards = np.zeros((args.buffer_size), dtype=np.float32)
    rb_next_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_dones = np.zeros((args.buffer_size), dtype=np.float32)
    
    rb_pos = 0
    rb_size = 0

    # 训练循环变量
    start_time = time.time()
    obs, _ = env.reset(seed=args.seed)
    
    # 用于平滑记录 train/reward
    recent_rewards = deque(maxlen=100)
    current_episode_reward = 0
    ema_reward = None

    print(f"Starting training for {args.total_timesteps} steps...")

    for global_step in range(args.total_timesteps):
        
        # 1. 动作选择
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = actor(obs_tensor)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                
                # 添加探索噪声
                noise = np.random.normal(0, env.action_space.high * args.exploration_noise, size=act_shape)
                action = (action + noise).clip(env.action_space.low, env.action_space.high)
                action = action.flatten()

        # 2. 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        current_episode_reward += reward

        # 记录详细指标 (每200步记录一次，避免日志过大，与 main.py 逻辑保持一致)
        if global_step % 200 == 0:
            writer.add_scalar('reward_terms/eta_0', float(info.get('eta_0', 0.0)), global_step)
            writer.add_scalar('reward_terms/comm_penalty', float(info.get('comm_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/eav_penalty', float(info.get('eav_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/energy_penalty', float(info.get('energy_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/boundary_penalty', float(info.get('boundary_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/action_smooth_penalty', float(info.get('action_smooth_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/reward_raw', float(info.get('reward_raw', 0.0)), global_step)
            writer.add_scalar('reward_terms/reward_final', float(info.get('reward_final', reward)), global_step)
            
            # 裁剪后指标
            writer.add_scalar('reward_terms/eta_0_clipped', float(info.get('eta_0_clipped', 0.0)), global_step)
            writer.add_scalar('reward_terms/comm_penalty_clipped', float(info.get('comm_penalty_clipped', 0.0)), global_step)
            writer.add_scalar('reward_terms/eav_penalty_clipped', float(info.get('eav_penalty_clipped', 0.0)), global_step)

        # 3. 存储经验
        real_next_obs = next_obs.copy() # 如果 truncated，此处在 Gym 中通常需要处理 info['final_observation']，但这里简化处理
        
        rb_obs[rb_pos] = obs
        rb_actions[rb_pos] = action
        rb_rewards[rb_pos] = reward
        rb_next_obs[rb_pos] = real_next_obs
        rb_dones[rb_pos] = done and not truncated # Truncated 不算真正的 done
        
        rb_pos = (rb_pos + 1) % args.buffer_size
        rb_size = min(rb_size + 1, args.buffer_size)

        obs = next_obs

        # 4. 回合结束处理
        if done or truncated:
            # 记录训练奖励
            recent_rewards.append(current_episode_reward)
            ema_reward = current_episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * current_episode_reward)
            
            writer.add_scalar('reward/train', current_episode_reward, global_step)
            writer.add_scalar('reward/train_ma100', np.mean(recent_rewards), global_step)
            writer.add_scalar('reward/train_ema', ema_reward, global_step)
            
            print(f"Step: {global_step}, Reward: {current_episode_reward:.2f}")
            
            obs, _ = env.reset(seed=args.seed + global_step) # 改变种子
            current_episode_reward = 0

        # 5. 训练模型
        if global_step > args.learning_starts:
            # 采样
            batch_inds = np.random.randint(0, rb_size, size=args.batch_size)
            
            b_obs = torch.tensor(rb_obs[batch_inds], dtype=torch.float32).to(device)
            b_actions = torch.tensor(rb_actions[batch_inds], dtype=torch.float32).to(device)
            b_rewards = torch.tensor(rb_rewards[batch_inds], dtype=torch.float32).to(device).unsqueeze(1)
            b_next_obs = torch.tensor(rb_next_obs[batch_inds], dtype=torch.float32).to(device)
            b_dones = torch.tensor(rb_dones[batch_inds], dtype=torch.float32).to(device).unsqueeze(1)

            # --- Critic 更新 ---
            with torch.no_grad():
                next_state_actions = target_actor(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs, next_state_actions)
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * qf1_next_target
            
            qf1_a_values = qf1(b_obs, b_actions)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            # --- Actor 更新 (延迟更新) ---
            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(b_obs, actor(b_obs)).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 软更新目标网络
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        # 6. 定期评估
        if global_step > 0 and global_step % args.eval_freq == 0:
            print(f"Evaluating at step {global_step}...")
            # 🔥 关键：同步归一化统计量
            sync_normalization_stats(env, eval_env)
            
            mean_ret, std_ret = evaluate(actor, eval_env, device, global_step)
            writer.add_scalar('reward/eval_mean', mean_ret, global_step)
            writer.add_scalar('reward/eval_std', std_ret, global_step)
            print(f"Evaluation result: {mean_ret:.2f} +/- {std_ret:.2f}")

    env.close()
    eval_env.close()
    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()