import argparse
import copy
from collections import deque
import time

import numpy as np
import torch

from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory, DiffusionMemory

from tensorboardX import SummaryWriter
import gymnasium as gym
import os
from logger import Logger
import datetime

# 修改1: 导入自定义环境
from myenv3 import UAVISACEnvironment
# CSV logging utility
from csv_logger import CSVExperimentLogger, create_scenario_name


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    v = str(v).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy for UAV-ISAC')
    
    parser.add_argument('--env_name', default="Env",
                        help='Custom UAV-ISAC environment (default: Env)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=2500000, metavar='N',
                        help='env timesteps (default: 2500000)')

    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='random exploration steps before policy (default: 10000)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, FlowMatching, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=20, metavar='N',
                        help='diffusion timesteps (default: 20)')
    
    parser.add_argument('--diffusion_lr', type=float, default=0.0001, metavar='G',
                        help='diffusion learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 2.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    parser.add_argument('--alpha_mean', type=float, default=0.001, metavar='G',
                        help='running mean update weight (default: 0.001)')

    parser.add_argument('--alpha_std', type=float, default=0.001, metavar='G',
                        help='running std update weight (default: 0.001)')

    parser.add_argument('--beta', type=float, default=1.0, metavar='G',
                        help='expQ weight (default: 1.0)')

    parser.add_argument('--weighted', action="store_true", help="weighted training")
    parser.add_argument('--aug', action="store_true", help="augmentation")

    parser.add_argument('--train_sample', type=int, default=64, metavar='N',
                        help='train_sample (default: 64)')

    parser.add_argument('--chosen', type=int, default=1, metavar='N', help="chosen actions (default:1)")
    parser.add_argument('--q_neg', type=float, default=0.0, metavar='G', help="q_neg (default: 0.0)")

    parser.add_argument('--behavior_sample', type=int, default=4, metavar='N', 
                        help="behavior_sample (default: 4)")
    parser.add_argument('--target_sample', type=int, default=4, metavar='N', 
                        help="target_sample (default: behavior sample)")

    parser.add_argument('--eval_sample', type=int, default=32, metavar='N', 
                        help="eval_sample (default: 32)")

    parser.add_argument('--deterministic', action="store_true", help="deterministic mode")

    parser.add_argument('--q_transform', type=str, default='qadv', metavar='S', 
                        help="q_transform (default: qadv)")

    parser.add_argument('--gradient', action="store_true", help="aug gradient")

    parser.add_argument('--policy_freq', type=int, default=1, metavar='N', 
                        help="policy_freq (default: 1)")

    parser.add_argument('--cut', type=float, default=1.0, metavar='G', help="cut (default: 1.0)")
    parser.add_argument('--times', type=int, default=1, metavar='N', help="times (default: 1)")

    parser.add_argument('--epsilon', type=float, default=0.0, metavar='G', 
                        help="eps greedy (default: 0.0)")
    
    parser.add_argument('--entropy_alpha', type=float, default=0.05, metavar='G', 
                        help="entropy_alpha (default: 0.05)")

    parser.add_argument('--eav_agg', type=str, default='top2', choices=['max', 'top2', 'logsumexp'],
                        help="eavesdropper SNR aggregation (default: logsumexp)")
    parser.add_argument('--eav_logsumexp_kappa', type=float, default=0.5, metavar='G',
                        help="logsumexp kappa for eav_agg=logsumexp (default: 0.5)")
    parser.add_argument('--eav_threshold', type=float, default=10.0, metavar='G',
                        help="eavesdropper threshold in dB (default: 10.0)")
    parser.add_argument('--eav_penalty_coef', type=float, default=3.0, metavar='G',
                        help="eavesdropper penalty coefficient (default: 3.0)")
    parser.add_argument('--eav_penalty_cap', type=float, default=20.0, metavar='G',
                        help="eavesdropper penalty cap (default: 20.0)")

    parser.add_argument('--comm_penalty', type=str, default='softplus', choices=['hinge', 'softplus', 'huber'],
                        help="communication penalty type (default: softplus)")
    parser.add_argument('--comm_threshold', type=float, default=10.0, metavar='G',
                        help="communication threshold in dB (default: 10.0)")
    parser.add_argument('--comm_penalty_coef', type=float, default=1.5, metavar='G',
                        help="communication penalty coefficient (default: 1.5)")
    parser.add_argument('--comm_softplus_kappa', type=float, default=5.0, metavar='G',
                        help="softplus kappa for comm_penalty=softplus (default: 5.0)")
    parser.add_argument('--comm_huber_delta', type=float, default=1.0, metavar='G',
                        help="huber delta for comm_penalty=huber (default: 1.0)")
    parser.add_argument('--comm_penalty_cap_per_user', type=float, default=15.0, metavar='G',
                        help="communication penalty cap per user (default: 15.0)")
    parser.add_argument('--comm_penalty_cap_total', type=float, default=30.0, metavar='G',
                        help="communication penalty cap total (default: 30.0)")
    parser.add_argument('--comm_penalty_avg_over_k', type=_str2bool, nargs='?', const=True, default=True,
                        help="average communication penalty over K users (default: True)")
    
    # 新增参数：动作平滑、用户移动范围、奖励缩放
    parser.add_argument('--action_smooth_coef', type=float, default=0.8, metavar='G',
                        help="action smoothness penalty coefficient (default: 0.8)")
    parser.add_argument('--user_move_range', type=float, default=20.0, metavar='G',
                        help="user movement range per step (default: 20.0)")
    parser.add_argument('--reward_scale', type=float, default=0.1, metavar='G',
                        help="reward scaling factor (default: 0.1)")
    
    # 建议3: 分项裁剪参数
    parser.add_argument('--eta_clip_max', type=float, default=15.0, metavar='G',
                        help="max clip value for sensing SNR (default: 15.0)")
    parser.add_argument('--comm_penalty_clip_max', type=float, default=5.0, metavar='G',
                        help="max clip value for comm penalty (default: 5.0)")
    parser.add_argument('--eav_penalty_clip_max', type=float, default=5.0, metavar='G',
                        help="max clip value for eav penalty (default: 5.0)")

    parser.add_argument('--load_id', type=str, default=None, metavar='S',
                        help="optional model id to load from ./results before training")
    
    # State preprocessing arguments
    if hasattr(argparse, 'BooleanOptionalAction'):
        parser.add_argument('--use_state_scaling', action=argparse.BooleanOptionalAction, default=True,
                            help="enable fixed state scaling in environment (default: True)")
        parser.add_argument('--use_obs_normalizer', action=argparse.BooleanOptionalAction, default=False,
                            help="enable agent-side Welford obs normalizer (default: False)")
    else:
        parser.add_argument('--use_state_scaling', type=_str2bool, nargs='?', const=True, default=True,
                            help="enable fixed state scaling in environment (default: True)")
        parser.add_argument('--use_obs_normalizer', type=_str2bool, nargs='?', const=True, default=False,
                            help="enable agent-side Welford obs normalizer (default: False)")
    
    parser.add_argument('--obs_norm_freeze_after', type=int, default=50000, metavar='N',
                        help="freeze obs normalizer stats after N steps (default: 50000)")
    parser.add_argument('--obs_norm_clip', type=float, default=5.0, metavar='G',
                        help="obs normalizer clip range (default: 5.0)")
    parser.add_argument('--obs_norm_eps', type=float, default=1e-8, metavar='G',
                        help="obs normalizer epsilon (default: 1e-8)")

    return parser.parse_args()


def evaluate(env, agent, steps, episodes=10):
    """评估函数 - 返回统一的评估结果字典
    
    Note: With fixed state scaling, train and eval environments use identical
    state preprocessing. No normalizer synchronization needed.
    """
    
    returns = np.zeros((episodes,), dtype=np.float32)
    
    # 评估泄露率统计
    eval_leakage_count = 0
    eval_total_users = 0
    
    # SNR统计 - 收集每个episode的平均值
    legal_snr_db_list = []  # 每个episode的合法接收器SNR平均值
    eav_snr_max_db_list = []  # 每个episode的最大窃听SNR平均值
    eav_snr_avg_db_list = []  # 每个episode的平均窃听SNR平均值
    
    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.
        done = False
        truncated = False
        
        # 单个episode内的SNR收集
        ep_legal_snr_list = []
        ep_eav_snr_max_list = []
        ep_eav_snr_avg_list = []
        
        while not (done or truncated):
            env_info = {
                'uav_x': env.uav_position[0],
                'uav_y': env.uav_position[1],
                'l_max': env.l_max,
                'x_min': env.X_min,
                'x_max': env.X_max,
                'y_min': env.Y_min,
                'y_max': env.Y_max,
            }
            action = agent.sample_action(state, eval=True, env_info=env_info)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            # 累计泄露率统计
            eval_leakage_count += info.get('leakage_count', 0)
            eval_total_users += info.get('total_users', 0)
            
            # 收集SNR信息（从info中获取）
            # 统计口径说明：
            # - legal_snr: 每个step的合法接收器感知SNR（标量）
            # - eav_snr_list: 每个step的K个窃听用户的SNR列表
            # - step_eav_snr_max: 该step下K个用户中的最大SNR
            # - step_eav_snr_avg: 该step下K个用户的平均SNR
            # 最终统计：对所有step求平均 -> episode平均 -> 多个episodes平均
            legal_snr = info.get('eta_0', np.nan)
            if not np.isnan(legal_snr):
                ep_legal_snr_list.append(legal_snr)
            
            # 收集窃听者SNR：每个step的最大值和平均值
            eav_snr_list = info.get('eavesdropper_snr_list', [])
            if len(eav_snr_list) > 0:
                ep_eav_snr_max_list.append(max(eav_snr_list))  # 该step的K个用户中的最大SNR
                ep_eav_snr_avg_list.append(np.mean(eav_snr_list))  # 该step的K个用户的平均SNR
        
        returns[i] = episode_reward
        
        # 计算该episode的SNR统计
        if len(ep_legal_snr_list) > 0:
            legal_snr_db_list.append(np.mean(ep_legal_snr_list))
        else:
            legal_snr_db_list.append(np.nan)
        
        # 计算窃听者SNR统计
        if len(ep_eav_snr_max_list) > 0:
            eav_snr_max_db_list.append(np.mean(ep_eav_snr_max_list))
        else:
            eav_snr_max_db_list.append(np.nan)
        
        if len(ep_eav_snr_avg_list) > 0:
            eav_snr_avg_db_list.append(np.mean(ep_eav_snr_avg_list))
        else:
            eav_snr_avg_db_list.append(np.nan)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 计算评估泄露率
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
    
    # 计算SNR gap
    if not np.isnan(legal_snr_db_mean) and not np.isnan(eav_snr_max_db_mean):
        snr_gap_db_mean = legal_snr_db_mean - eav_snr_max_db_mean
    else:
        snr_gap_db_mean = np.nan
    
    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'reward: {mean_return:<5.1f}  '
          f'std: {std_return:<5.1f}  '
          f'leakage_rate: {eval_leakage_rate:.2%}')
    print(returns)
    print('-' * 60)
    
    # 返回统一的评估结果字典
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


def main(args=None, logger=None, id=None):

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir = "record"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', 
                          f'ratio={args.ratio}', f'seed={args.seed}')
    if id is not None:
        log_dir = os.path.join(log_dir, f'run_id={id}')
    writer = SummaryWriter(log_dir)
    
    # CSV logging initialization
    csv_dir = os.path.join(log_dir, 'csv_logs')
    scenario_name = create_scenario_name(args)
    run_id = id if id is not None else datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    csv_logger = CSVExperimentLogger(
        run_id=run_id,
        algorithm='QVPO',
        seed=args.seed,
        scenario_name=scenario_name,
        eval_interval=10000,  # Will be set below
        csv_dir=csv_dir
    )
    
    # Training start time for CSV logging
    training_start_time = time.time()

    # 🔥🔥🔥 关键修改：直接实例化环境，传入归一化参数
    print("Initializing UAV-ISAC Environment...")
    print(f"  - State scaling (fixed): {args.use_state_scaling}")
    print(f"  - Agent-side obs normalizer: {args.use_obs_normalizer}")
    if args.use_obs_normalizer:
        print(f"    * Freeze after: {args.obs_norm_freeze_after} steps")
        print(f"    * Clip range: {args.obs_norm_clip}")
        print(f"    * Epsilon: {args.obs_norm_eps}")
    
    env = UAVISACEnvironment(
        use_state_scaling=args.use_state_scaling,
        eav_threshold=args.eav_threshold,
        eav_penalty_coef=args.eav_penalty_coef,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
        comm_threshold=args.comm_threshold,
        comm_penalty_coef=args.comm_penalty_coef,
        comm_softplus_kappa=args.comm_softplus_kappa,
        comm_penalty_clip_per_user=args.comm_penalty_cap_per_user,
        comm_penalty_clip_total=args.comm_penalty_cap_total,
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
        reward_scale=args.reward_scale,
    )
    
    eval_env = UAVISACEnvironment(
        use_state_scaling=args.use_state_scaling,
        eav_threshold=args.eav_threshold,
        eav_penalty_coef=args.eav_penalty_coef,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
        comm_threshold=args.comm_threshold,
        comm_penalty_coef=args.comm_penalty_coef,
        comm_softplus_kappa=args.comm_softplus_kappa,
        comm_penalty_clip_per_user=args.comm_penalty_cap_per_user,
        comm_penalty_clip_total=args.comm_penalty_cap_total,
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
        reward_scale=args.reward_scale,
    )
    
    # 获取状态和动作维度
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 训练参数
    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = args.start_steps
    eval_interval = 10000
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10

    recent_rewards = deque(maxlen=100)
    ema_reward = None
    
    # Episode-level leakage rate tracking (MA100)
    recent_leakage_rates = deque(maxlen=100)

    # 创建经验池
    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    # 创建QVPO智能体
    print("Creating QVPO agent...")
    agent = QVPO(args, state_size, env.action_space, memory, diffusion_memory, device)

    if args.load_id is not None:
        agent.load_model(os.path.join('./results', prefix + '_' + name), id=args.load_id)

    steps = 0
    episodes = 0
    best_result = -float('inf')
    best_step = 0  # Track step where best result was achieved

    print(f"Starting training for {num_steps} steps...")
    print(f"Random exploration for first {start_steps} steps")

    # 泄露率统计变量
    # Global cumulative (renamed to avoid confusion)
    train_leakage_count_global = 0
    train_total_users_global = 0
    
    # Window-based (200 steps)
    window_leakage_count = 0
    window_total_users = 0
    
    # 训练奖励跟踪：记录最近一个已完成episode的总reward
    # 用于CSV日志的train_reward字段，语义明确为"最近完成的训练episode总reward"
    last_completed_episode_reward = np.nan

    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        truncated = False
        
        # Episode-level leakage tracking
        episode_leakage_count = 0
        episode_total_users = 0
        
        state, _ = env.reset(seed=args.seed + episodes)
        episodes += 1
        
        while not (done or truncated):
            # 动作选择
            if start_steps > steps:
                action = env.action_space.sample()
            else:
                env_info = {
                    'uav_x': env.uav_position[0],
                    'uav_y': env.uav_position[1],
                    'l_max': env.l_max,
                    'x_min': env.X_min,
                    'x_max': env.X_max,
                    'y_min': env.Y_min,
                    'y_max': env.Y_max,
                }
                action = agent.sample_action(state, eval=False, env_info=env_info)
            
            next_state, reward, done, truncated, info = env.step(action)

            # 累计泄露率统计
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

            if steps % 200 == 0:
                writer.add_scalar('reward_terms/eta_0', float(info.get('eta_0', 0.0)), steps)
                writer.add_scalar('reward_terms/comm_penalty', float(info.get('comm_penalty', 0.0)), steps)
                writer.add_scalar('reward_terms/eav_penalty', float(info.get('eav_penalty', 0.0)), steps)
                writer.add_scalar('reward_terms/energy_penalty', float(info.get('energy_penalty', 0.0)), steps)
                writer.add_scalar('reward_terms/boundary_penalty', float(info.get('boundary_penalty', 0.0)), steps)
                writer.add_scalar('reward_terms/action_smooth_penalty', float(info.get('action_smooth_penalty', 0.0)), steps)
                writer.add_scalar('reward_terms/reward_raw', float(info.get('reward_raw', 0.0)), steps)
                writer.add_scalar('reward_terms/reward_clip_1', float(info.get('reward_clip_1', reward)), steps)
                writer.add_scalar('reward_terms/reward_final', float(info.get('reward_final', reward)), steps)
                # 建议3: 分项裁剪后的值
                writer.add_scalar('reward_terms/eta_0_clipped', float(info.get('eta_0_clipped', 0.0)), steps)
                writer.add_scalar('reward_terms/comm_penalty_clipped', float(info.get('comm_penalty_clipped', 0.0)), steps)
                writer.add_scalar('reward_terms/eav_penalty_clipped', float(info.get('eav_penalty_clipped', 0.0)), steps)
                
                # 感知泄漏率相关指标
                step_leakage_count = info.get('leakage_count', 0)
                step_total_users = info.get('total_users', 0)
                if step_total_users > 0:
                    step_leakage_rate = step_leakage_count / step_total_users
                    writer.add_scalar('security/step_leakage_rate', step_leakage_rate, steps)
                writer.add_scalar('security/step_leakage_count', float(step_leakage_count), steps)
                writer.add_scalar('security/eav_penalty_raw', float(info.get('eav_penalty_raw', 0.0)), steps)
                writer.add_scalar('security/eav_penalty_weighted', float(info.get('eav_penalty_weighted', 0.0)), steps)
                
                # Window-based leakage rate (200 steps)
                if window_total_users > 0:
                    window_leakage_rate = window_leakage_count / window_total_users
                    writer.add_scalar('security/train_leakage_rate_window200', window_leakage_rate, steps)
                else:
                    # Handle zero denominator: write 0.0 when no users in window
                    writer.add_scalar('security/train_leakage_rate_window200', 0.0, steps)
                
                # Reset window counters after logging
                window_leakage_count = 0
                window_total_users = 0
                
                # Optional: Global cumulative leakage rate (renamed to avoid confusion)
                if train_total_users_global > 0:
                    train_leakage_rate_global = train_leakage_count_global / train_total_users_global
                    writer.add_scalar('security/train_leakage_rate_global', train_leakage_rate_global, steps)

            # mask计算
            mask = 0.0 if (done or truncated) else args.gamma

            steps += 1
            episode_steps += 1
            episode_reward += reward

            # 存储经验
            agent.append_memory(state, action, reward, next_state, mask)

            # 训练更新
            if steps >= start_steps:
                agent.train(steps, updates_per_step, batch_size=batch_size, log_writer=writer)
                # 熵系数退火
                agent.entropy_alpha = min(args.entropy_alpha, 
                                         max(0.002, args.entropy_alpha - steps/num_steps * args.entropy_alpha))

            # 定期评估
            if steps % eval_interval == 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {steps}")
                print(f"{'='*60}")
                eval_results = evaluate(eval_env, agent, steps, episodes=10)
                
                # TensorBoard logging (保持原有逻辑)
                writer.add_scalar('reward/eval_mean', eval_results['mean_return'], steps)
                writer.add_scalar('security/eval_leakage_rate', eval_results['eval_leakage_rate'], steps)
                
                # CSV logging for training metrics
                time_elapsed = time.time() - training_start_time
                # train_reward: 最近一个已完成训练episode的总reward（不是当前进行中的episode）
                # 如果还没有完成任何episode，则为NaN
                train_reward_ma = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else np.nan
                
                csv_logger.log_training_metrics(
                    eval_results=eval_results,
                    step=steps,
                    time_elapsed_sec=time_elapsed,
                    train_reward=last_completed_episode_reward,
                    train_reward_ma100=train_reward_ma
                )
                
                if eval_results['mean_return'] > best_result:
                    best_result = eval_results['mean_return']
                    best_step = steps
                    print(f"New best result: {best_result:.2f}! Saving model...")
                    agent.save_model(os.path.join('./results', prefix + '_' + name), id=id)

            state = next_state

        # Episode结束：更新last_completed_episode_reward
        # 这个值将用于CSV日志的train_reward字段
        last_completed_episode_reward = episode_reward
        
        # Episode-level leakage rate (MA100)
        if episode_total_users > 0:
            episode_leakage_rate = episode_leakage_count / episode_total_users
            recent_leakage_rates.append(episode_leakage_rate)
        # Note: if episode_total_users == 0, we skip appending to avoid NaN in the deque
        
        # 记录episode结束时的reward日志
        recent_rewards.append(episode_reward)
        ema_reward = episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * episode_reward)

        writer.add_scalar('reward/train', episode_reward, steps)
        writer.add_scalar('reward/train_ma100', float(np.mean(recent_rewards)), steps)
        writer.add_scalar('reward/train_ema', float(ema_reward), steps)
        
        # Episode-level leakage rate MA100
        if len(recent_leakage_rates) > 0:
            writer.add_scalar('security/train_leakage_rate_ma100', float(np.mean(recent_leakage_rates)), steps)

        if episodes % log_interval == 0:
            pass

        print(f'Episode: {episodes:<4}  '
              f'Steps: {episode_steps:<4}  '
              f'Total Steps: {steps:<7}  '
              f'Reward: {episode_reward:<5.1f}')

        if logger is not None:
            for i in range(episode_steps):
                logger.add(epoch=steps-episode_steps+i, reward=episode_reward)

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
    
    final_eval_results = evaluate(eval_env, agent, steps, episodes=10)
    training_total_time = time.time() - training_start_time
    
    # CSV logging for final comparison
    csv_logger.log_final_comparison(
        final_eval_results=final_eval_results,
        total_train_steps=steps,
        best_eval_reward=best_result,
        best_step=best_step,
        training_time_sec=training_total_time
    )
    
    print(f"Total episodes: {episodes}")
    print(f"Total steps: {steps}")
    print(f"Best evaluation result: {best_result:.2f} at step {best_step}")
    print(f"Final evaluation result: {final_eval_results['mean_return']:.2f}")
    print(f"Final leakage rate: {final_eval_results['eval_leakage_rate']:.2%}")
    print(f"Training time: {training_total_time:.2f} seconds")
    print(f"CSV logs saved to: {csv_dir}")
    print(f"{'='*60}")
    
    # 关闭环境和writer
    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    args = readParser()
    if args.target_sample == -1:
        args.target_sample = args.behavior_sample

    ## 设置
    prefix = 'qvpo'
    name = args.env_name
    keys = ("epoch", "reward")
    times = args.times
    id = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    
    print(f"\n{'#'*60}")
    print(f"# QVPO Training Configuration")
    print(f"# Seed: {args.seed}")
    print(f"# Total steps: {args.num_steps}")
    print(f"# Run ID: {id}")
    print(f"{'#'*60}\n")
    
    # 创建结果目录
    result_dir = os.path.join('./results', prefix + '_' + name)
    os.makedirs(result_dir, exist_ok=True)
    
    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps)+2100, 
                   times=times, config=args, path=result_dir, id=id)

    ## 运行训练
    for run_idx in range(times):
        print(f"\n{'#'*60}")
        print(f"# Starting training run {run_idx+1}/{times}")
        print(f"{'#'*60}\n")
        main(args, logger=logger, id=id+"_"+str(run_idx))

    logger.save(result_dir, id=id)
    print("\nAll training runs completed!")