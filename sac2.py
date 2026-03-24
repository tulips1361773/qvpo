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
# CSV logging utility
from csv_logger import CSVExperimentLogger, create_scenario_name

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

def evaluate_sac(env, actor, obs_rms, device, episodes=10):
    """
    SAC evaluation function - returns unified evaluation results dict.
    Matches the structure of QVPO's evaluate() function.
    """
    actor.eval()
    returns = np.zeros((episodes,), dtype=np.float32)
    
    # Leakage rate statistics
    eval_leakage_count = 0
    eval_total_users = 0
    
    # SNR statistics - collect average per episode
    legal_snr_db_list = []
    eav_snr_max_db_list = []
    eav_snr_avg_db_list = []
    
    for i in range(episodes):
        eval_obs, _ = env.reset()
        eval_done = False
        eval_episode_ret = 0.0
        
        # Per-episode SNR collection
        ep_legal_snr_list = []
        ep_eav_snr_max_list = []
        ep_eav_snr_avg_list = []
        
        while not eval_done:
            # Normalize observation using training statistics
            norm_eval_obs = obs_rms.normalize(eval_obs)
            
            with torch.no_grad():
                # Get deterministic action (mean)
                _, _, eval_action_mean = actor.get_action(
                    torch.Tensor(norm_eval_obs).to(device).unsqueeze(0)
                )
                eval_action = eval_action_mean.cpu().numpy().flatten()
            
            eval_next_obs, eval_r, eval_term, eval_trunc, eval_info = env.step(eval_action)
            eval_episode_ret += eval_r
            eval_done = eval_term or eval_trunc
            eval_obs = eval_next_obs
            
            # Accumulate leakage rate statistics
            eval_leakage_count += eval_info.get('leakage_count', 0)
            eval_total_users += eval_info.get('total_users', 0)
            
            # 收集SNR信息（从info中获取）
            # 统计口径说明：
            # - legal_snr: 每个step的合法接收器感知SNR（标量）
            # - eav_snr_list: 每个step的K个窃听用户的SNR列表
            # - step_eav_snr_max: 该step下K个用户中的最大SNR
            # - step_eav_snr_avg: 该step下K个用户的平均SNR
            # 最终统计：对所有step求平均 -> episode平均 -> 多个episodes平均
            legal_snr = eval_info.get('eta_0', np.nan)
            if not np.isnan(legal_snr):
                ep_legal_snr_list.append(legal_snr)
            
            # 收集窃听者SNR：每个step的最大值和平均值
            eav_snr_list = eval_info.get('eavesdropper_snr_list', [])
            if len(eav_snr_list) > 0:
                ep_eav_snr_max_list.append(max(eav_snr_list))  # 该step的K个用户中的最大SNR
                ep_eav_snr_avg_list.append(np.mean(eav_snr_list))  # 该step的K个用户的平均SNR
        
        returns[i] = eval_episode_ret
        
        # Calculate per-episode SNR statistics
        if len(ep_legal_snr_list) > 0:
            legal_snr_db_list.append(np.mean(ep_legal_snr_list))
        else:
            legal_snr_db_list.append(np.nan)
        
        # Calculate eavesdropper SNR statistics
        if len(ep_eav_snr_max_list) > 0:
            eav_snr_max_db_list.append(np.mean(ep_eav_snr_max_list))
        else:
            eav_snr_max_db_list.append(np.nan)
        
        if len(ep_eav_snr_avg_list) > 0:
            eav_snr_avg_db_list.append(np.mean(ep_eav_snr_avg_list))
        else:
            eav_snr_avg_db_list.append(np.nan)
    
    actor.train()
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Calculate evaluation leakage rate
    eval_leakage_rate = eval_leakage_count / eval_total_users if eval_total_users > 0 else 0.0
    
    # 计算SNR统计（跨多个evaluation episodes的平均）
    # 统计口径说明（平均安全暴露水平）：
    # - legal_snr_db_mean: 所有eval episodes中所有steps的合法接收器SNR的平均值
    # - eav_snr_max_db_mean: 所有eval episodes中，每个step的"K个用户最大SNR"的平均值
    # - eav_snr_avg_db_mean: 所有eval episodes中，每个step的"K个用户平均SNR"的平均值
    # 注意：这是"平均安全暴露水平"，不是"全局最坏时刻最大SNR"
    legal_snr_db_mean = np.nanmean(legal_snr_db_list) if len(legal_snr_db_list) > 0 else np.nan
    legal_snr_db_std = np.nanstd(legal_snr_db_list) if len(legal_snr_db_list) > 0 else np.nan
    eav_snr_max_db_mean = np.nanmean(eav_snr_max_db_list) if len(eav_snr_max_db_list) > 0 else np.nan
    eav_snr_avg_db_mean = np.nanmean(eav_snr_avg_db_list) if len(eav_snr_avg_db_list) > 0 else np.nan
    
    # Calculate SNR gap
    if not np.isnan(legal_snr_db_mean) and not np.isnan(eav_snr_max_db_mean):
        snr_gap_db_mean = legal_snr_db_mean - eav_snr_max_db_mean
    else:
        snr_gap_db_mean = np.nan
    
    # Print evaluation results (matching QVPO format)
    print('-' * 60)
    print(f'reward: {mean_return:<5.1f}  '
          f'std: {std_return:<5.1f}  '
          f'leakage_rate: {eval_leakage_rate:.2%}')
    print(returns)
    print('-' * 60)
    
    # Return unified evaluation results dict
    eval_results = {
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'eval_leakage_rate': float(eval_leakage_rate),
        'legal_snr_db_mean': float(legal_snr_db_mean),
        'legal_snr_db_std': float(legal_snr_db_std),
        'eav_snr_max_db_mean': float(eav_snr_max_db_mean),
        'eav_snr_avg_db_mean': float(eav_snr_avg_db_mean),
        'snr_gap_db_mean': float(snr_gap_db_mean),
        'eval_episode_count': episodes,
    }
    
    return eval_results

if __name__ == "__main__":
    args = parse_args()
    
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"sac_seed{args.seed}_{current_time_str}"
    log_dir = os.path.join("record", "sac", f"{args.exp_name}_{current_time_str}")
    writer = SummaryWriter(log_dir)
    
    # CSV logging initialization
    csv_dir = os.path.join(log_dir, 'csv_logs')
    scenario_name = create_scenario_name(args)
    csv_logger = CSVExperimentLogger(
        run_id=run_id,
        algorithm='SAC',
        seed=args.seed,
        scenario_name=scenario_name,
        eval_interval=args.eval_frequency,
        csv_dir=csv_dir
    )
    
    print(f"\n{'#'*60}")
    print(f"# SAC Training Configuration")
    print(f"# Seed: {args.seed}")
    print(f"# Total steps: {args.total_timesteps}")
    print(f"# Run ID: {run_id}")
    print(f"{'#'*60}\n")

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
    
    # Episode-level leakage rate tracking (MA100)
    recent_leakage_rates = deque(maxlen=100)
    
    # 泄露率统计（对齐main.py）
    # Global cumulative (renamed to avoid confusion)
    train_leakage_count_global = 0
    train_total_users_global = 0
    
    # Window-based (200 steps)
    window_leakage_count = 0
    window_total_users = 0
    
    # Best result tracking for CSV logging
    best_eval_reward = -float('inf')
    best_step = 0
    
    # Training start time for CSV logging
    training_start_time = time.time()
    
    # 训练奖励跟踪：记录最近一个已完成episode的总reward
    # 用于CSV日志的train_reward字段，语义明确为"最近完成的训练episode总reward"
    last_completed_episode_reward = np.nan
    
    obs, _ = env.reset(seed=args.seed)
    # 初始状态更新到归一化器
    obs_rms.update(obs)

    episode_reward = 0
    episode_steps = 0
    episodes_count = 0
    
    # Episode-level leakage tracking
    episode_leakage_count = 0
    episode_total_users = 0
    
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
        step_leakage = info.get('leakage_count', 0)
        step_users = info.get('total_users', 0)
        
        # Global cumulative (for reference only)
        train_leakage_count_global += step_leakage
        train_total_users_global += step_users
        
        # Window-based (200 steps)
        window_leakage_count += step_leakage
        window_total_users += step_users
        
        # Episode-level tracking
        episode_leakage_count += step_leakage
        episode_total_users += step_users
        
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
            
            # Window-based leakage rate (200 steps)
            if window_total_users > 0:
                window_leakage_rate = window_leakage_count / window_total_users
                writer.add_scalar('security/train_leakage_rate_window200', window_leakage_rate, global_step)
            else:
                # Handle zero denominator: write 0.0 when no users in window
                writer.add_scalar('security/train_leakage_rate_window200', 0.0, global_step)
            
            # Reset window counters after logging
            window_leakage_count = 0
            window_total_users = 0
            
            # Optional: Global cumulative leakage rate (renamed to avoid confusion)
            if train_total_users_global > 0:
                train_leakage_rate_global = train_leakage_count_global / train_total_users_global
                writer.add_scalar('security/train_leakage_rate_global', train_leakage_rate_global, global_step)

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

        # Episode 结束
        done = terminated or truncated
        if done:
            episodes_count += 1
            
            # Episode结束：更新last_completed_episode_reward
            # 这个值将用于CSV日志的train_reward字段
            last_completed_episode_reward = episode_reward
            
            # Episode-level leakage rate (MA100)
            if episode_total_users > 0:
                episode_leakage_rate = episode_leakage_count / episode_total_users
                recent_leakage_rates.append(episode_leakage_rate)
            # Note: if episode_total_users == 0, we skip appending to avoid NaN in the deque
            
            recent_rewards.append(episode_reward)
            ema_reward = episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * episode_reward)
            
            # TensorBoard 记录
            writer.add_scalar('reward/train', episode_reward, global_step)
            writer.add_scalar('reward/train_ma100', float(np.mean(recent_rewards)), global_step)
            writer.add_scalar('reward/train_ema', float(ema_reward), global_step)
            
            # Episode-level leakage rate MA100
            if len(recent_leakage_rates) > 0:
                writer.add_scalar('security/train_leakage_rate_ma100', float(np.mean(recent_leakage_rates)), global_step)
            
            print(f'Episode: {episodes_count:<4}  '
                  f'Steps: {episode_steps:<4}  '
                  f'Global Step: {global_step:<7}  '
                  f'Reward: {episode_reward:<5.1f}')
            
            # Reset episode
            obs, _ = env.reset()
            obs_rms.update(obs)
            episode_reward = 0
            episode_steps = 0
            
            # Reset episode-level leakage tracking
            episode_leakage_count = 0
            episode_total_users = 0

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
            print(f"\n{'='*60}")
            print(f"Evaluation at step {global_step}")
            print(f"{'='*60}")
            
            # Use unified evaluate function
            eval_results = evaluate_sac(eval_env, actor, obs_rms, device, episodes=args.eval_episodes)
            
            # TensorBoard logging (保持原有逻辑)
            writer.add_scalar('reward/eval_mean', eval_results['mean_return'], global_step)
            writer.add_scalar('security/eval_leakage_rate', eval_results['eval_leakage_rate'], global_step)
            
            # CSV logging for training metrics
            time_elapsed = time.time() - training_start_time
            # train_reward: 最近一个已完成训练episode的总reward（不是当前进行中的episode）
            # 如果还没有完成任何episode，则为NaN
            train_reward_ma = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else np.nan
            
            csv_logger.log_training_metrics(
                eval_results=eval_results,
                step=global_step,
                time_elapsed_sec=time_elapsed,
                train_reward=last_completed_episode_reward,
                train_reward_ma100=train_reward_ma
            )
            
            # Track best result
            if eval_results['mean_return'] > best_eval_reward:
                best_eval_reward = eval_results['mean_return']
                best_step = global_step
                print(f"New best result: {best_eval_reward:.2f}!")

    # ============================================================
    # 训练结束 - 执行最终独立评估
    # ============================================================
    # 说明：
    # 1. 这是训练结束后的独立评估，用于统一记录final_comparison CSV
    # 2. 即使最后一个训练step恰好也触发了常规evaluate，仍保留此final evaluate
    # 3. 这样可以保证最终结果记录流程一致，final_comparison中的数据以此为准
    # 4. 不依赖最后一次中间评估的结果，确保final数据的独立性和可重复性
    print(f"\n{'='*60}")
    print(f"Training completed! Performing final evaluation...")
    print(f"{'='*60}")
    
    final_eval_results = evaluate_sac(eval_env, actor, obs_rms, device, episodes=args.eval_episodes)
    training_total_time = time.time() - training_start_time
    
    # CSV logging for final comparison
    csv_logger.log_final_comparison(
        final_eval_results=final_eval_results,
        total_train_steps=global_step,
        best_eval_reward=best_eval_reward,
        best_step=best_step,
        training_time_sec=training_total_time
    )
    
    print(f"Total steps: {global_step}")
    print(f"Best evaluation result: {best_eval_reward:.2f} at step {best_step}")
    print(f"Final evaluation result: {final_eval_results['mean_return']:.2f}")
    print(f"Final leakage rate: {final_eval_results['eval_leakage_rate']:.2%}")
    print(f"Training time: {training_total_time:.2f} seconds")
    print(f"CSV logs saved to: {csv_dir}")
    print(f"{'='*60}")
    
    env.close()
    eval_env.close()
    writer.close()