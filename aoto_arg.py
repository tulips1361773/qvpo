import argparse
import copy
from collections import deque
import numpy as np
import torch
import gymnasium as gym
import os
import datetime
from tensorboardX import SummaryWriter

# 引入 Agent
from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory, DiffusionMemory
from logger import Logger

# 引入自定义环境
from myenv3 import UAVISACEnvironment

def _str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return True
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy for UAV-ISAC')
    
    # --- 1. 环境基础参数 ---
    parser.add_argument('--env_name', default="Env", help='Custom UAV-ISAC environment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_steps', type=int, default=2000000, help='env timesteps')
    parser.add_argument('--start_steps', type=int, default=10000, help='random exploration steps')

    # --- 2. 算法核心参数 (QVPO/Diffusion) ---
    # 丢失的参数主要是这一块
    parser.add_argument('--cuda', default='cuda:0', help='run on CUDA')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='target smoothing coefficient')
    parser.add_argument('--update_actor_target_every', type=int, default=1, help='update actor target freq')
    parser.add_argument("--policy_type", type=str, default="Diffusion", help="Diffusion, FlowMatching, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=20, help='diffusion timesteps')
    parser.add_argument('--diffusion_lr', type=float, default=0.0001, help='diffusion learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.0003, help='critic learning rate')
    parser.add_argument('--action_lr', type=float, default=0.03, help='action learning rate')
    
    # 🔥 报错缺失的参数在这里 🔥
    parser.add_argument('--noise_ratio', type=float, default=1.0, help='noise ratio')
    parser.add_argument('--action_gradient_steps', type=int, default=20, help='action gradient steps')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio of action grad norm')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, help='grad norm')
    parser.add_argument('--alpha_mean', type=float, default=0.001, help='running mean update')
    parser.add_argument('--alpha_std', type=float, default=0.001, help='running std update')
    parser.add_argument('--beta', type=float, default=1.0, help='expQ weight')
    
    # 采样与增强参数
    parser.add_argument('--weighted', action="store_true", help="weighted training")
    parser.add_argument('--aug', action="store_true", help="augmentation")
    parser.add_argument('--train_sample', type=int, default=64, help='train_sample')
    parser.add_argument('--chosen', type=int, default=1, help="chosen actions")
    parser.add_argument('--q_neg', type=float, default=0.0, help="q_neg")
    parser.add_argument('--behavior_sample', type=int, default=4, help="behavior_sample")
    parser.add_argument('--target_sample', type=int, default=4, help="target_sample")
    parser.add_argument('--eval_sample', type=int, default=32, help="eval_sample")
    parser.add_argument('--deterministic', action="store_true", help="deterministic mode")
    parser.add_argument('--q_transform', type=str, default='qadv', help="q_transform")
    parser.add_argument('--gradient', action="store_true", help="aug gradient")
    parser.add_argument('--policy_freq', type=int, default=1, help="policy_freq")
    parser.add_argument('--cut', type=float, default=1.0, help="cut")
    parser.add_argument('--times', type=int, default=1, help="times")
    parser.add_argument('--epsilon', type=float, default=0.0, help="eps greedy")
    parser.add_argument('--entropy_alpha', type=float, default=0.05, help="entropy_alpha")

    # --- 3. 归一化开关 ---
    if hasattr(argparse, 'BooleanOptionalAction'):
        parser.add_argument('--normalize_state', action=argparse.BooleanOptionalAction, default=True, help="state norm")
    else:
        parser.add_argument('--normalize_state', type=_str2bool, nargs='?', const=True, default=True, help="state norm")

    # --- 4. 物理/环境参数 (将被自动校准覆盖) ---
    parser.add_argument('--reward_scale', type=float, default=0.1)
    
    # 核心阈值
    parser.add_argument('--comm_threshold', type=float, default=10.0)
    parser.add_argument('--eav_threshold', type=float, default=10.0)
    
    # 惩罚系数 (Auto-Tune 会修改这些)
    parser.add_argument('--comm_penalty_coef', type=float, default=0.5)
    parser.add_argument('--eav_penalty_coef', type=float, default=2.0)
    
    # 裁剪与平滑 (部分参数可能为了兼容旧代码保留)
    parser.add_argument('--comm_penalty_clip_per_user', type=float, default=20.0)
    parser.add_argument('--comm_penalty_clip_total', type=float, default=50.0)
    parser.add_argument('--eav_penalty_clip_max', type=float, default=1000.0)
    parser.add_argument('--comm_softplus_kappa', type=float, default=2.0)
    
    # 其他动作参数
    parser.add_argument('--action_smooth_coef', type=float, default=0.8)
    parser.add_argument('--user_move_range', type=float, default=20.0)
    
    # 负载
    parser.add_argument('--load_id', type=str, default=None, help="model id to load")

    # 旧代码兼容参数 (保留以防其他地方调用)
    parser.add_argument('--eav_agg', type=str, default='logsumexp')
    parser.add_argument('--eav_logsumexp_kappa', type=float, default=0.5)
    parser.add_argument('--eav_penalty_cap', type=float, default=20.0)
    parser.add_argument('--comm_penalty', type=str, default='softplus')
    parser.add_argument('--comm_huber_delta', type=float, default=1.0)
    parser.add_argument('--comm_penalty_cap_per_user', type=float, default=15.0)
    parser.add_argument('--comm_penalty_cap_total', type=float, default=30.0)
    parser.add_argument('--comm_penalty_avg_over_k', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--comm_penalty_clip_max', type=float, default=5.0)

    return parser.parse_args()

# ==============================================================================
# 🔥 核心功能：物理计算与自动校准
# ==============================================================================
def softplus(x, kappa=2.0):
    return np.logaddexp(0, kappa * x) / kappa

def calc_physics_raw(env, uav_pos, user_positions, power_alloc):
    """计算原始物理值(dB)"""
    # 1. Eta
    d_t = np.linalg.norm(uav_pos - env.target_position)
    d_r = np.linalg.norm(env.target_position - env.radar_receiver_position)
    G_tx, G_rx, c, fc = 13, 13, 3e8, 2.4e9
    lambda_c = c / fc
    radar_const = (10**(G_tx/10) * 10**(G_rx/10) * lambda_c**2) / ((4 * np.pi)**3)
    P_r = (power_alloc * radar_const) / (max(d_t**2, 1e-5) * max(d_r**2, 1e-5))
    eta_db = 10 * np.log10(max(P_r / env.sigma2, 1e-10))
    
    # 2. Comm (简化计算用于评估)
    comm_snrs = []
    H = env.H
    for k in range(env.K):
        d_2d = np.linalg.norm(uav_pos[:2] - user_positions[k][:2])
        d_3d = np.sqrt(H**2 + d_2d**2)
        # 简化的LOS/NLOS平均模型近似
        path_loss_db = 20*np.log10(d_3d) + 20*np.log10(fc) - 147.55 + 2 # 粗略估计
        snr_db = 10*np.log10(power_alloc) - path_loss_db - 10*np.log10(env.sigma2)
        # 这里为了校准准确性，直接调用 env 的函数更准
        snr_db = env._calculate_communication_snr(d_2d, power_alloc)
        comm_snrs.append(snr_db)
        
    # 3. Eav
    eav_snrs = env._calculate_sensing_snr_eavesdropper(uav_pos, power_alloc)
        
    return eta_db, np.array(comm_snrs), np.array(eav_snrs)

def auto_tune_params(args):
    print("\n" + "="*60)
    print("🚀 运行自动校准 (Auto-Calibration) - 优化版")
    print("   原则：放宽惩罚斜率，避免初期梯度爆炸导致Agent放弃任务。")
    
    env = UAVISACEnvironment(normalize_state=False)
    stats_eta = []
    
    # 1. 采样获取感知奖励的分布
    print("   [1/3] 采样环境以确定感知奖励基准...")
    for _ in range(50):
        env.reset()
        done = False
        while not done:
            if np.random.rand() < 0.4:
                action = env.action_space.sample()
            else:
                # 简单的飞向目标策略
                target_vec = env.target_position[:2] - env.uav_position[:2]
                dist = np.linalg.norm(target_vec)
                angle = np.arctan2(target_vec[1], target_vec[0]) / np.pi 
                action = np.array([angle, 1.0, 1.0], dtype=np.float32)
            _, _, done, _, _ = env.step(action)
            p_val = (action[2] + 1)/2 * env.P_max
            eta = env._calculate_sensing_snr_legal(env.uav_position, p_val)
            stats_eta.append(eta)
    env.close()
    
    # 2. 确定基准值 R_max
    if len(stats_eta) > 0:
        p95_eta = np.percentile(stats_eta, 95)
        R_max = max(p95_eta, 5.0)
    else:
        R_max = 20.0
    print(f"   --> 最大期望感知收益 (R_max): {R_max:.2f} dB")

    # 3. 设定惩罚系数 (优化点)
    # 原逻辑：delta=1.0, beta=1.5 (太陡峭)
    # 新逻辑：delta=3.0, beta=1.0 (违规3dB时，惩罚等于最大奖励)
    # 这样给 Agent 一个“软缓冲区”，允许轻微违规，但严重违规依然会被重罚
    
    delta_bad_sec = 3.0  # 放宽：容忍 3dB 的过渡带
    beta_sec = 1.2       # 惩罚倍率：违规 3dB 时，惩罚是 R_max 的 1.2 倍
    
    delta_bad_comm = 3.0 
    beta_comm = 0.5      
    
    # 计算 Eav 系数
    # Formula: coef * softplus(delta) = beta * R_max
    raw_val_sec = softplus(delta_bad_sec, kappa=2.0)
    args.eav_penalty_coef = round((R_max * beta_sec) / raw_val_sec, 2)
    
    # 计算 Comm 系数
    raw_val_comm = softplus(delta_bad_comm, kappa=2.0)
    args.comm_penalty_coef = round((R_max * beta_comm) / raw_val_comm, 2)
    
    # 4. 设定 Reward Scale
    # 稍微调小一点 Scale，让数值更稳定
    args.reward_scale = round(1.0 / R_max, 4)
    
    print(f"   [2/3] 校准结果:")
    print(f"   -- 安全系数 (Sec Coef): {args.eav_penalty_coef} (Target: Violate {delta_bad_sec}dB -> Pen {beta_sec}xReward)")
    print(f"   -- 通信系数 (Comm Coef): {args.comm_penalty_coef}")
    print(f"   -- 奖励缩放 (Reward Scale): {args.reward_scale}")
    print("="*60 + "\n")
    return args

def evaluate(env, agent, steps, source_env=None, args=None):
    """评估函数：增加详细的奖励成分分解与物理行为监控"""
    if source_env and hasattr(source_env, 'state_normalizer'):
        env.state_normalizer.mean = source_env.state_normalizer.mean.copy()
        env.state_normalizer.var = source_env.state_normalizer.var.copy()
        env.state_normalizer.count = source_env.state_normalizer.count
        env.state_normalizer.set_training(False)
    
    episodes = 10
    returns = np.zeros(episodes)
    total_steps = 0
    
    # --- 新增：详细统计累加器 ---
    metrics = {
        'leakage_count': 0,
        'max_violation': 0.0,
        'accum_eta': 0.0,          # 累积感知 SNR (正收益来源)
        'accum_sec_pen': 0.0,      # 累积安全惩罚 (权重后)
        'accum_comm_pen': 0.0,     # 累积通信惩罚 (权重后)
        'accum_power': 0.0,        # 累积发射功率 (0-1比率)
        'accum_dist': 0.0          # 累积与目标距离
    }
    
    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.
        done = False
        
        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            state = next_state
            total_steps += 1
            
            # --- 1. 物理违规统计 ---
            p_val = (action[2] + 1)/2 * env.P_max
            # 重新计算物理值用于统计 (或者直接信赖 info 中的值，这里重新计算更稳妥)
            _, _, eavs = calc_physics_raw(env, env.uav_position, env.user_positions, p_val)
            max_eav = np.max(eavs)
            gap = max_eav - args.eav_threshold
            
            if gap > 0:
                metrics['leakage_count'] += 1
                if gap > metrics['max_violation']:
                    metrics['max_violation'] = gap
            
            # --- 2. 奖励成分拆解 (从 info 获取) ---
            # myenv3.py 的 info 中包含: eta_0, eav_penalty_weighted, comm_penalty (unweighted raw avg)
            # 注意: 这里需要确保 myenv3 返回的 comm_penalty 是我们想要的。
            # 如果 myenv3 中是 R_comm = avg_clipped * coef，info['comm_penalty'] 最好是最终扣的分
            
            metrics['accum_eta'] += info.get('eta_0', 0.0)
            metrics['accum_sec_pen'] += info.get('eav_penalty_weighted', 0.0)
            
            # 手动计算一下加权后的通信惩罚，或者假设 info 里的是原始值
            # 假设 info['comm_penalty'] 是原始平均惩罚，我们需要乘系数
            raw_comm = info.get('comm_penalty', 0.0)
            # 由于 myenv3 中可能有截断，这里近似计算
            metrics['accum_comm_pen'] += raw_comm * args.comm_penalty_coef
            
            # --- 3. 动作行为统计 ---
            # action[1] 是距离: [-1, 1] -> [0, l_max] ? 不对，step里是 action[1] * l_max
            # 让我们直接看物理距离
            dist_to_target = np.linalg.norm(env.uav_position - env.target_position)
            metrics['accum_dist'] += dist_to_target
            metrics['accum_power'] += (action[2] + 1) / 2  # 0.0 - 1.0
        
        returns[i] = episode_reward
    
    if hasattr(env, 'state_normalizer'):
        env.state_normalizer.set_training(True)
    
    mean_return = np.mean(returns)
    
    # --- 计算平均值 ---
    avg_leakage = metrics['leakage_count'] / max(total_steps, 1)
    avg_eta = metrics['accum_eta'] / max(total_steps, 1)
    avg_sec_pen = metrics['accum_sec_pen'] / max(total_steps, 1)
    avg_comm_pen = metrics['accum_comm_pen'] / max(total_steps, 1)
    avg_power = metrics['accum_power'] / max(total_steps, 1)
    avg_dist = metrics['accum_dist'] / max(total_steps, 1)
    
    print('-' * 60)
    print(f'Eval Step {steps} | Ret: {mean_return:.2f}')
    print(f'   [Gain] Eta (Sense): {avg_eta:.2f} dB')
    print(f'   [Loss] Sec Pen: {avg_sec_pen:.2f} | Comm Pen: {avg_comm_pen:.2f}')
    print(f'   [Phys] Power: {avg_power:.2%} | Dist: {avg_dist:.1f}m')
    print(f'   [Stat] Leakage: {avg_leakage:.1%} | Max Viol: {metrics["max_violation"]:.2f} dB')
    print('-' * 60)
    
    return {
        'mean_return': mean_return,
        'leakage_rate': avg_leakage,
        'max_violation': metrics['max_violation'],
        'avg_eta': avg_eta,
        'avg_sec_pen': avg_sec_pen,
        'avg_comm_pen': avg_comm_pen,
        'avg_power': avg_power,
        'avg_dist': avg_dist
    }

def main(args=None, logger=None, id=None):
    # 执行自动校准
    args = auto_tune_params(args)
    
    dir = "record"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', f'seed={args.seed}')
    if id: log_dir = os.path.join(log_dir, f'run_id={id}')
    writer = SummaryWriter(log_dir)

    # 实例化环境
    env_kwargs = {
        'normalize_state': args.normalize_state,
        'comm_threshold': args.comm_threshold,
        'eav_threshold': args.eav_threshold,
        'comm_penalty_coef': args.comm_penalty_coef,
        'eav_penalty_coef': args.eav_penalty_coef,
        'reward_scale': args.reward_scale,
        'comm_penalty_clip_total': args.comm_penalty_clip_total,
        'eav_penalty_clip_max': 1e5,
    }
    
    env = UAVISACEnvironment(**env_kwargs)
    eval_env = UAVISACEnvironment(**env_kwargs)
    
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    memory = ReplayMemory(state_size, action_size, 1000000, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, 1000000, device)
    agent = QVPO(args, state_size, env.action_space, memory, diffusion_memory, device)
    
    steps = 0
    episodes = 0
    best_result = -float('inf')
    
    print("Starting Training...")
    
    while steps < args.num_steps:
        state, _ = env.reset(seed=args.seed + episodes)
        done = False
        ep_reward = 0
        ep_steps = 0
        
        # 训练过程中的物理统计 (用于填充 Outcome 图表)
        train_sec_outages = 0
        train_comm_outages = 0
        episodes += 1
        
        while not done:
            if args.start_steps > steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, eval=False)
            
            next_state, reward, done, _, info = env.step(action)
            steps += 1
            ep_steps += 1
            ep_reward += reward
            
            # --- 统计训练时的物理违规情况 ---
            p_val = (action[2] + 1)/2 * env.P_max
            _, comms, eavs = calc_physics_raw(env, env.uav_position, env.user_positions, p_val)
            
            if np.any(eavs > args.eav_threshold):
                train_sec_outages += 1
            if np.any(comms < args.comm_threshold):
                train_comm_outages += 1
            # ------------------------------------

            mask = 0.0 if done else args.gamma
            agent.append_memory(state, action, reward, next_state, mask)
            
            if steps >= args.start_steps:
                agent.train(steps, 1, args.batch_size, writer)
                
            if steps % 200 == 0:
                writer.add_scalar('reward_terms/weighted_eav_penalty', info.get('eav_penalty_weighted', 0), steps)
                writer.add_scalar('reward_terms/raw_eta', info.get('eta_0', 0), steps)

    # 在 while steps < args.num_steps 循环内:
    
            if steps % 5000 == 0:
                # 调用修改后的 evaluate，它现在返回一个字典
                eval_metrics = evaluate(eval_env, agent, steps, env, args)
                
                # 记录原来的指标
                writer.add_scalar('reward/eval_mean', eval_metrics['mean_return'], steps)
                writer.add_scalar('Outcome/Eval_Sensing_Leakage_Rate', eval_metrics['leakage_rate'], steps)
                writer.add_scalar('Outcome/Eval_Max_Secrecy_Violation_dB', eval_metrics['max_violation'], steps)
                
                # 🔥 新增：详细诊断指标 🔥
                # 1. 奖励成分分析：看谁在拖后腿
                writer.add_scalar('Diagnosis/Avg_Sensing_Gain_dB', eval_metrics['avg_eta'], steps)
                writer.add_scalar('Diagnosis/Avg_Secrecy_Penalty', eval_metrics['avg_sec_pen'], steps)
                writer.add_scalar('Diagnosis/Avg_Comm_Penalty', eval_metrics['avg_comm_pen'], steps)
                
                # 2. 行为分析：Agent 是不是怂了（功率低）或者跑了（距离远）
                writer.add_scalar('Diagnosis/Action_Avg_Power_Ratio', eval_metrics['avg_power'], steps)
                writer.add_scalar('Diagnosis/Action_Avg_Distance_m', eval_metrics['avg_dist'], steps)
                
                 # 新增几何分析图表
                writer.add_scalar('Diagnosis/Avg_Min_User_Distance', eval_metrics['avg_min_user_dist'], steps)
                writer.add_scalar('Diagnosis/Favorable_Geometry_Ratio', eval_metrics['favorable_ratio'], steps)
                
                if eval_metrics['mean_return'] > best_result:
                    best_result = eval_metrics['mean_return']
                    agent.save_model(os.path.join('./results', f'{args.env_name}'), id)
            
            state = next_state
            
        # 记录 Episode 级别的统计
        writer.add_scalar('reward/train', ep_reward, steps)
        
        # 记录训练过程中的 Outage Rates (解决空图问题)
        train_sec_rate = train_sec_outages / max(ep_steps, 1)
        train_comm_rate = train_comm_outages / max(ep_steps, 1)
        writer.add_scalar('Outcome/Train_Secrecy_Outage_Rate', train_sec_rate, steps)
        writer.add_scalar('Outcome/Train_Comm_Outage_Rate', train_comm_rate, steps)

    env.close()
    writer.close()

if __name__ == "__main__":
    args = readParser()
    if args.target_sample == -1: args.target_sample = args.behavior_sample
    
    # 简单日志封装
    name = args.env_name
    id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    logger = Logger(name=name, keys=("epoch", "reward"), max_epochs=args.num_steps, config=args, path='./results', id=id)
    
    main(args, logger, id)