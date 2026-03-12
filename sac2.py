import argparse
import os
import random
import time
import datetime
from distutils.util import strtobool
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 🔥 修改：导入 myenv3 环境
from myenv3 import UAVISACEnvironment

# ============================================================
# Agent 端状态归一化 (替代环境内部的归一化)
# ============================================================
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, clip=10.0):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.clip = clip

    def update(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x):
        # (x - mean) / std
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    # ==========================================
    # 实验基础设置
    # ==========================================
    parser.add_argument("--exp-name", type=str, default="sac_uav", help="experiment name")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, help="deterministic torch")
    parser.add_argument('--cuda', default='cuda:0', help='run on CUDA')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, help="wandb tracking")
    
    # ==========================================
    # SAC 算法参数
    # ==========================================
    parser.add_argument("--total-timesteps", type=int, default=2500000, help="total timesteps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update coef")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--learning-starts", type=int, default=10000, help="timesteps before learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="policy learning rate")
    parser.add_argument("--q-lr", type=float, default=1e-3, help="q network learning rate")
    parser.add_argument("--policy-frequency", type=int, default=2, help="policy update freq")
    parser.add_argument("--target-network-frequency", type=int, default=1, help="target network update freq")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy coef")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, help="auto alpha")

    # ==========================================
    # 评估参数
    # ==========================================
    parser.add_argument("--eval-frequency", type=int, default=10000, help="evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, help="number of episodes for evaluation")

    # ==========================================
    # 环境参数 (完全对齐 main.py，确保物理模型一致)
    # ==========================================
    # 注意：sac使用agent端归一化，所以这里normalize_state虽然传入但env内部应忽略或设为False
    parser.add_argument('--normalize_state', type=lambda x: bool(strtobool(str(x))), default=False, 
                        help="Should be False for SAC env, handled by Agent")
    
    # 奖励缩放与动作平滑
    parser.add_argument('--reward_scale', type=float, default=0.1, help="reward scaling factor")
    parser.add_argument('--action_smooth_coef', type=float, default=0.8, help="action smoothness penalty coefficient")
    parser.add_argument('--user_move_range', type=float, default=20.0, help="user movement range per step")

    # 窃听 (Eavesdropper) 相关 (适配myenv3)
    parser.add_argument('--eav_threshold', type=float, default=10.0)
    parser.add_argument('--eav_penalty_coef', type=float, default=2.0)
    parser.add_argument('--eav_penalty_clip_max', type=float, default=1000.0)

    # 通信 (Communication) 相关 (适配myenv3)
    parser.add_argument('--comm_threshold', type=float, default=10.0)
    parser.add_argument('--comm_penalty_coef', type=float, default=0.5)
    parser.add_argument('--comm_softplus_kappa', type=float, default=2.0)
    parser.add_argument('--comm_penalty_clip_per_user', type=float, default=20.0)
    parser.add_argument('--comm_penalty_clip_total', type=float, default=50.0)

    args = parser.parse_args()
    return args

def make_uav_env(args):
    """
    初始化环境，传入所有物理参数 (适配myenv3)
    注意：SAC不启用StateNormalizer，由Agent端进行归一化
    """
    env = UAVISACEnvironment(
        # 核心参数
        normalize_state=False,  # SAC使用Agent端归一化，禁用环境内部归一化
        normalize_reward=True,
        
        # 窃听相关
        eav_threshold=args.eav_threshold,
        eav_penalty_coef=args.eav_penalty_coef,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
        
        # 通信相关
        comm_threshold=args.comm_threshold,
        comm_penalty_coef=args.comm_penalty_coef,
        comm_softplus_kappa=args.comm_softplus_kappa,
        comm_penalty_clip_per_user=args.comm_penalty_clip_per_user,
        comm_penalty_clip_total=args.comm_penalty_clip_total,
        
        # 其他
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
        reward_scale=args.reward_scale,
    )
    return env

# ============================================================
# 网络定义
# ============================================================
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = -5 + 0.5 * (2 - (-5)) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

if __name__ == "__main__":
    args = parse_args()
    
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("record", "sac", f"{args.exp_name}_{current_time_str}")
    writer = SummaryWriter(log_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 设备设置 - 支持指定GPU编号
    if args.cuda.startswith("cuda"):
        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 环境初始化
    print("Initializing Environments (Using myenv3 without internal normalization)...")
    env = make_uav_env(args)
    eval_env = make_uav_env(args)

    # 🔥 Agent 端状态归一化初始化
    # 因为环境是Raw的，所以Agent必须维护一个归一化器
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Buffer 初始化
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    rb_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_actions = np.zeros((args.buffer_size, *act_shape), dtype=np.float32)
    rb_rewards = np.zeros((args.buffer_size), dtype=np.float32)
    rb_next_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_dones = np.zeros((args.buffer_size), dtype=np.float32)
    rb_ptr = 0
    rb_size = 0

    # 统计相关
    recent_rewards = deque(maxlen=100)
    ema_reward = None
    
    # 泄露率统计（对齐main.py）
    train_leakage_count = 0
    train_total_users = 0
    
    obs, _ = env.reset(seed=args.seed)
    # 初始状态更新到归一化器
    obs_rms.update(obs)

    episode_reward = 0
    episode_steps = 0
    episodes_count = 0
    
    print("Starting training...")

    for global_step in range(args.total_timesteps):
        
        # 1. 动作选择
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            # 🔥 归一化后输入网络
            norm_obs = obs_rms.normalize(obs)
            with torch.no_grad():
                action, _, _ = actor.get_action(torch.Tensor(norm_obs).to(device).unsqueeze(0))
                action = action.cpu().numpy().flatten()

        # 2. 环境步进 (获取 Raw State)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        
        # 累计泄露率统计（对齐main.py）
        train_leakage_count += info.get('leakage_count', 0)
        train_total_users += info.get('total_users', 0)
        
        # 🔥 完全对齐main.py的日志记录（每200步）
        if global_step % 200 == 0:
            # 奖励项（对齐main.py）
            writer.add_scalar('reward_terms/eta_0', float(info.get('eta_0', 0.0)), global_step)
            writer.add_scalar('reward_terms/comm_penalty', float(info.get('comm_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/eav_penalty', float(info.get('eav_penalty_raw', 0.0)), global_step)  # 对齐main.py使用原始值
            writer.add_scalar('reward_terms/energy_penalty', float(info.get('energy_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/boundary_penalty', float(info.get('boundary_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/action_smooth_penalty', float(info.get('action_smooth_penalty', 0.0)), global_step)
            writer.add_scalar('reward_terms/reward_raw', float(info.get('reward_raw', 0.0)), global_step)
            writer.add_scalar('reward_terms/reward_clip_1', float(info.get('reward_final', reward)), global_step)  # 对齐main.py的命名
            writer.add_scalar('reward_terms/reward_final', float(info.get('reward_final', reward)), global_step)
            
            # 裁剪值（使用环境提供的真实值）
            writer.add_scalar('reward_terms/eta_0_clipped', float(info.get('eta_0_clipped', 0.0)), global_step)
            writer.add_scalar('reward_terms/comm_penalty_clipped', float(info.get('comm_penalty_clipped', 0.0)), global_step)
            writer.add_scalar('reward_terms/eav_penalty_clipped', float(info.get('eav_penalty_clipped', 0.0)), global_step)
            
            # 感知泄漏率相关指标（对齐main.py）
            step_leakage_count = info.get('leakage_count', 0)
            step_total_users = info.get('total_users', 0)
            if step_total_users > 0:
                step_leakage_rate = step_leakage_count / step_total_users
                writer.add_scalar('security/step_leakage_rate', step_leakage_rate, global_step)
            writer.add_scalar('security/step_leakage_count', float(step_leakage_count), global_step)
            writer.add_scalar('security/eav_penalty_raw', float(info.get('eav_penalty_raw', 0.0)), global_step)
            writer.add_scalar('security/eav_penalty_weighted', float(info.get('eav_penalty_weighted', 0.0)), global_step)
            
            # 训练泄露率（每200步记录一次累计泄露率）
            if train_total_users > 0:
                train_leakage_rate = train_leakage_count / train_total_users
                writer.add_scalar('security/train_leakage_rate', train_leakage_rate, global_step)

        # 3. Buffer 存储 (存 Raw Data)
        real_done = terminated 
        rb_obs[rb_ptr] = obs
        rb_actions[rb_ptr] = action
        rb_rewards[rb_ptr] = reward
        rb_next_obs[rb_ptr] = next_obs
        rb_dones[rb_ptr] = real_done
        rb_ptr = (rb_ptr + 1) % args.buffer_size
        rb_size = min(rb_size + 1, args.buffer_size)

        episode_reward += reward
        obs = next_obs
        
        # 更新归一化统计量 (用新遇到的 Raw State)
        obs_rms.update(obs)

        # 4. 回合结束处理
        if terminated or truncated:
            episodes_count += 1
            
            recent_rewards.append(episode_reward)
            ema_reward = episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * episode_reward)
            
            # 🔥 对齐日志：使用与 main.py 一致的标签
            writer.add_scalar('reward/train', episode_reward, global_step)
            writer.add_scalar('reward/train_ma100', float(np.mean(recent_rewards)), global_step)
            writer.add_scalar('reward/train_ema', float(ema_reward), global_step)
            
            if episodes_count % 10 == 0:
                print(f"Step: {global_step}, Episode: {episodes_count}, Reward: {episode_reward:.2f}")

            obs, _ = env.reset(seed=None)
            obs_rms.update(obs) # Reset 后也要 update
            episode_reward = 0
            episode_steps = 0

        # 5. 训练逻辑
        if global_step > args.learning_starts:
            idxs = np.random.randint(0, rb_size, size=args.batch_size)
            
            # 从 Buffer 取出 Raw Data
            b_obs_raw = rb_obs[idxs]
            b_next_obs_raw = rb_next_obs[idxs]
            
            # 🔥 关键：在训练前实时归一化 batch
            # 这解决了异策略 Buffer 数据分布漂移的问题
            b_obs = torch.tensor(obs_rms.normalize(b_obs_raw), device=device)
            b_next_obs = torch.tensor(obs_rms.normalize(b_next_obs_raw), device=device)
            
            b_actions = torch.tensor(rb_actions[idxs], device=device)
            b_rewards = torch.tensor(rb_rewards[idxs], device=device)
            b_dones = torch.tensor(rb_dones[idxs], device=device)

            # 更新 Critic
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs, next_state_actions)
                qf2_next_target = qf2_target(b_next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * args.gamma * (min_qf_next_target.view(-1))

            qf1_a_values = qf1(b_obs, b_actions).view(-1)
            qf2_a_values = qf2(b_obs, b_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # 更新 Actor
            if global_step % args.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(b_obs)
                qf1_pi = qf1(b_obs, pi)
                qf2_pi = qf2(b_obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(b_obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # 🔥 记录SAC算法特定的指标（每1000步记录一次，避免过多日志）
            if global_step % 1000 == 0:
                writer.add_scalar('losses/qf_loss', qf_loss.item(), global_step)
                writer.add_scalar('losses/actor_loss', actor_loss.item(), global_step)
                if args.autotune:
                    writer.add_scalar('alpha/value', alpha, global_step)
                    writer.add_scalar('losses/alpha_loss', alpha_loss.item(), global_step)

        # ============================================================
        # 🔥 评估逻辑 (完全对齐 main.py)
        # ============================================================
        if global_step > 0 and global_step % args.eval_frequency == 0:
            print(f"Evaluating at step {global_step}")
            print(f"{'='*60}")
            
            # SAC评估时不需要同步Env内部的归一化器(因为Env是Raw的)
            # 而是直接使用 obs_rms 对 eval_env 的输出进行归一化
            
            actor.eval()
            returns = np.zeros((args.eval_episodes,), dtype=np.float32)
            
            # 评估泄露率统计（对齐main.py）
            eval_leakage_count = 0
            eval_total_users = 0
            
            for i in range(args.eval_episodes):
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_episode_ret = 0.0
                
                while not eval_done:
                    # 🔥 使用训练得到的统计量归一化评估状态
                    norm_eval_obs = obs_rms.normalize(eval_obs)
                    
                    with torch.no_grad():
                        # 获取确定性动作 (mean)
                        _, _, eval_action_mean = actor.get_action(
                            torch.Tensor(norm_eval_obs).to(device).unsqueeze(0)
                        )
                        eval_action = eval_action_mean.cpu().numpy().flatten()
                    
                    eval_next_obs, eval_r, eval_term, eval_trunc, eval_info = eval_env.step(eval_action)
                    eval_episode_ret += eval_r
                    eval_done = eval_term or eval_trunc
                    eval_obs = eval_next_obs
                    
                    # 累计泄露率统计（对齐main.py）
                    eval_leakage_count += eval_info.get('leakage_count', 0)
                    eval_total_users += eval_info.get('total_users', 0)
                
                returns[i] = eval_episode_ret
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 计算评估泄露率（对齐main.py）
            eval_leakage_rate = eval_leakage_count / eval_total_users if eval_total_users > 0 else 0.0
            
            # 🔥 记录测试时的指标 (与 main.py 完全一致)
            writer.add_scalar('reward/eval_mean', mean_return, global_step)
            writer.add_scalar('security/eval_leakage_rate', eval_leakage_rate, global_step)
            
            # 打印格式对齐main.py
            print('-' * 60)
            print(f'Num steps: {global_step:<5}  '
                  f'reward: {mean_return:<5.1f}  '
                  f'std: {std_return:<5.1f}  '
                  f'leakage_rate: {eval_leakage_rate:.2%}')
            print(returns)
            print('-' * 60)
            
            actor.train()

    env.close()
    eval_env.close()
    writer.close()