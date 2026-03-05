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
    
    # --- 环境基础参数 ---
    parser.add_argument('--env_name', default="Env", help='Custom UAV-ISAC environment')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_steps', type=int, default=2500000, help='env timesteps')
    parser.add_argument('--start_steps', type=int, default=10000, help='random exploration steps')

    # --- 算法参数 (QVPO/Diffusion) ---
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
    parser.add_argument('--noise_ratio', type=float, default=1.0, help='noise ratio')
    parser.add_argument('--action_gradient_steps', type=int, default=20, help='action gradient steps')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio of action grad norm')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, help='grad norm')
    parser.add_argument('--cuda', default='cuda:0', help='run on CUDA')
    parser.add_argument('--alpha_mean', type=float, default=0.001, help='running mean update')
    parser.add_argument('--alpha_std', type=float, default=0.001, help='running std update')
    parser.add_argument('--beta', type=float, default=1.0, help='expQ weight')
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

    # --- 归一化开关 ---
    if hasattr(argparse, 'BooleanOptionalAction'):
        parser.add_argument('--normalize_state', action=argparse.BooleanOptionalAction, default=True, help="state norm")
    else:
        parser.add_argument('--normalize_state', type=_str2bool, nargs='?', const=True, default=True, help="state norm")

    # --- 奖励函数配置 (将被自动校准覆盖，这里只是初始占位) ---
    parser.add_argument('--reward_scale', type=float, default=0.1)
    
    # 核心阈值
    parser.add_argument('--comm_threshold', type=float, default=10.0)
    parser.add_argument('--eav_threshold', type=float, default=10.0)
    
    # 裁剪与系数 (Auto-Tune 会修改这些)
    parser.add_argument('--comm_penalty_clip_per_user', type=float, default=10.0)
    parser.add_argument('--comm_penalty_clip_total', type=float, default=15.0)
    parser.add_argument('--comm_penalty_coef', type=float, default=0.5)
    
    parser.add_argument('--eav_penalty_clip_max', type=float, default=10.0)
    parser.add_argument('--eav_penalty_coef', type=float, default=2.0)
    
    # 其他
    parser.add_argument('--action_smooth_coef', type=float, default=0.8)
    parser.add_argument('--user_move_range', type=float, default=20.0)
    
    # 负载
    parser.add_argument('--load_id', type=str, default=None, help="model id to load")

    # 为了兼容旧代码可能存在的参数调用，保留一些占位符
    parser.add_argument('--eav_agg', type=str, default='logsumexp')
    parser.add_argument('--eav_logsumexp_kappa', type=float, default=0.5)
    parser.add_argument('--eav_penalty_cap', type=float, default=20.0)
    parser.add_argument('--comm_penalty', type=str, default='softplus')
    parser.add_argument('--comm_softplus_kappa', type=float, default=5.0)
    parser.add_argument('--comm_huber_delta', type=float, default=1.0)
    parser.add_argument('--comm_penalty_cap_per_user', type=float, default=15.0)
    parser.add_argument('--comm_penalty_cap_total', type=float, default=30.0)
    parser.add_argument('--comm_penalty_avg_over_k', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--comm_penalty_clip_max', type=float, default=5.0) # Old param placeholder

    return parser.parse_args()


# ==============================================================================
# 🔥 核心功能：自动物理校准与参数统计 (已修正版)
# ==============================================================================
def softplus(x):
    return np.logaddexp(0, x)

def calc_physics_raw(env, uav_pos, user_positions, power_alloc):
    """在 main.py 中复刻物理公式，用于提取未裁剪的原始值"""
    # 1. 感知 SNR (Eta)
    d_t = np.linalg.norm(uav_pos - env.target_position)
    d_r = np.linalg.norm(env.target_position - env.radar_receiver_position)
    # 这里的常数必须与 myenv.py 保持一致
    G_tx, G_rx = 13, 13
    c, fc = 3e8, 2.4e9
    lambda_c = c / fc
    sigma = 1.0
    radar_const = (10**(G_tx/10) * 10**(G_rx/10) * lambda_c**2 * sigma) / ((4 * np.pi)**3)
    
    P_r = (power_alloc * radar_const) / (max(d_t**2, 1e-10) * max(d_r**2, 1e-10))
    eta_linear = P_r / env.sigma2
    eta_db = 10 * np.log10(max(eta_linear, 1e-10))
    
    # 2. 通信 SNR
    comm_snrs = []
    H = env.H
    for k in range(env.K):
        dist_2d = np.linalg.norm(uav_pos[:2] - user_positions[k][:2])
        d_3d = np.sqrt(H**2 + dist_2d**2)
        theta = np.arcsin(H / d_3d) * 180 / np.pi
        c1, c2 = 12.081, 0.11395
        p_los = 1 / (1 + c1 * np.exp(-c2 * (theta - c1)))
        mu_los, mu_nlos = 1.44544, 199.526
        alpha, K_0 = 2.0, (4 * np.pi * fc) / c
        pl_los = mu_los * (K_0 * d_3d)**alpha
        pl_nlos = mu_nlos * (K_0 * d_3d)**alpha
        # 假设环境中使用 dB 加权或线性几何平均，这里简化复刻 dB 逻辑
        L_db = p_los * 10*np.log10(pl_los) + (1 - p_los) * 10*np.log10(pl_nlos)
        L_lin = 10**(L_db/10)
        snr = ((1.0/L_lin) * power_alloc) / env.sigma2
        comm_snrs.append(10 * np.log10(max(snr, 1e-10)))
        
    # 3. 窃听 SNR
    eav_snrs = []
    for k in range(env.K):
        d_kr = np.linalg.norm(env.target_position - user_positions[k])
        P_rk = (power_alloc * radar_const) / (max(d_t**2, 1e-10) * max(d_kr**2, 1e-10))
        eav_snrs.append(10 * np.log10(max(P_rk/env.sigma2, 1e-10)))
        
    return eta_db, np.array(comm_snrs), np.array(eav_snrs)

def auto_tune_params(args):
    """运行少量仿真，自动设定合适的裁剪值和系数 (修复版)"""
    print("\n" + "="*60)
    print("🚀 正在运行自动校准 (Auto-Calibration)...")
    print("   目的：统计物理数值分布，以设定最佳的 Clip 和 Scale 值")
    
    env = UAVISACEnvironment(normalize_state=False)
    
    stats = {'eta': [], 'comm_pen_raw': [], 'eav_pen_raw': []}
    num_episodes = 50
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if np.random.rand() < 0.5:
                action = env.action_space.sample()
            else:
                target_vec = env.target_position[:2] - env.uav_position[:2]
                dist = np.linalg.norm(target_vec)
                if dist > 1.0: target_vec = target_vec / dist
                angle = np.arctan2(target_vec[1], target_vec[0]) / np.pi 
                angle += np.random.normal(0, 0.1)
                action = np.array([angle, 1.0, 1.0], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)
            
            _, _, done, _, _ = env.step(action)
            p_val = (action[2] + 1)/2 * env.P_max
            eta, comms, eavs = calc_physics_raw(env, env.uav_position, env.user_positions, p_val)
            
            stats['eta'].append(eta)
            
            gaps = args.comm_threshold - comms
            pens = softplus(gaps)
            valid_pens = pens[pens > 0.01]
            if len(valid_pens) > 0:
                stats['comm_pen_raw'].extend(valid_pens)
                
            m = np.max(eavs)
            agg = m + np.log(np.sum(np.exp(eavs - m)))
            gap_e = agg - args.eav_threshold
            pen_e = softplus(gap_e)
            if pen_e > 0.01:
                stats['eav_pen_raw'].append(pen_e)
                
    env.close()
    
    eta_arr = np.array(stats['eta'])
    comm_arr = np.array(stats['comm_pen_raw'])
    eav_arr = np.array(stats['eav_pen_raw'])
    
    if len(eta_arr) > 0:
        p95_eta = np.percentile(eta_arr, 95)
        ref_max_reward = max(p95_eta, 5.0)
    else:
        ref_max_reward = 20.0
        
    print(f"   [统计] 感知SNR (Eta) P95: {ref_max_reward:.2f}")

    if len(comm_arr) > 0:
        p95_comm = np.percentile(comm_arr, 95)
        args.comm_penalty_clip_per_user = float(np.ceil(p95_comm))
        args.comm_penalty_clip_total = float(np.ceil(args.comm_penalty_clip_per_user * 1.5))
    else:
        args.comm_penalty_clip_per_user = 10.0
        args.comm_penalty_clip_total = 15.0

    if len(eav_arr) > 0:
        p95_eav = np.percentile(eav_arr, 95)
        args.eav_penalty_clip_max = float(np.ceil(p95_eav))
    else:
        args.eav_penalty_clip_max = 10.0
        
    raw_comm_coef = ref_max_reward / max(args.comm_penalty_clip_total, 1.0)
    args.comm_penalty_coef = round(raw_comm_coef, 1)
    
    raw_eav_coef = ref_max_reward / max(args.eav_penalty_clip_max, 1.0)
    args.eav_penalty_coef = round(raw_eav_coef, 1)
    
    est_max_pen_total = (args.comm_penalty_clip_total * args.comm_penalty_coef) + \
                        (args.eav_penalty_clip_max * args.eav_penalty_coef)
    
    est_range = ref_max_reward + est_max_pen_total
    args.reward_scale = round(2.0 / est_range, 3)

    print(f"✅ 校准完成! 参数已更新:")
    print(f"   --eav_penalty_clip_max: {args.eav_penalty_clip_max}")
    print(f"   --comm_penalty_clip_per_user: {args.comm_penalty_clip_per_user}")
    print(f"   --comm_penalty_clip_total: {args.comm_penalty_clip_total}")
    print(f"   --comm_penalty_coef: {args.comm_penalty_coef} (Target ~{ref_max_reward:.1f})")
    print(f"   --eav_penalty_clip_max: {args.eav_penalty_clip_max}")
    print(f"   --eav_penalty_coef: {args.eav_penalty_coef} (Target ~{ref_max_reward:.1f})")
    print(f"   --reward_scale: {args.reward_scale}")
    print("="*60 + "\n")
    return args


def evaluate(env, agent, steps, source_env=None, args=None):
    """评估函数：增加物理中断率统计"""
    if source_env is not None and hasattr(source_env, 'state_normalizer') and hasattr(env, 'state_normalizer'):
        env.state_normalizer.mean = source_env.state_normalizer.mean.copy()
        env.state_normalizer.var = source_env.state_normalizer.var.copy()
        env.state_normalizer.count = source_env.state_normalizer.count

    if hasattr(env, 'state_normalizer'):
        env.state_normalizer.set_training(False)
    
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)
    
    # 统计计数器
    total_eval_steps = 0
    total_comm_outages = 0
    total_secrecy_outages = 0
    
    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            # --- 物理性能统计 ---
            total_eval_steps += 1
            # 还原功率 P = (act + 1)/2 * P_max
            p_val = (action[2] + 1)/2 * env.P_max
            # 计算真实物理值
            _, comms, eavs = calc_physics_raw(env, env.uav_position, env.user_positions, p_val)
            
            # 1. 通信中断判定: 任意用户 SNR < 阈值 即视为系统中断
            #    注意: args.comm_threshold 是 dB 值
            if np.any(comms < args.comm_threshold):
                total_comm_outages += 1
                
            # 2. 窃听判定: 任意用户被窃听 SNR > 阈值 即视为泄密
            if np.any(eavs > args.eav_threshold):
                total_secrecy_outages += 1
        
        returns[i] = episode_reward
    
    if hasattr(env, 'state_normalizer'):
        env.state_normalizer.set_training(True)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 计算概率
    comm_outage_prob = total_comm_outages / max(total_eval_steps, 1)
    secrecy_outage_prob = total_secrecy_outages / max(total_eval_steps, 1)
    
    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'Return: {mean_return:<5.1f}  '
          f'Comm Outage: {comm_outage_prob:.2%}  '
          f'Secrecy Outage: {secrecy_outage_prob:.2%}')
    print(returns)
    print('-' * 60)
    return mean_return, comm_outage_prob, secrecy_outage_prob


def main(args=None, logger=None, id=None):
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 🔥🔥 第一步：执行自动参数校准 🔥🔥
    args = auto_tune_params(args)

    dir = "record"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', 
                          f'ratio={args.ratio}', f'seed={args.seed}')
    if id is not None:
        log_dir = os.path.join(log_dir, f'run_id={id}')
    writer = SummaryWriter(log_dir)

    print("Initializing UAV-ISAC Environment (Training)...")
    
    # 🔥🔥 第二步：使用校准后的参数实例化环境 🔥🔥
    env = UAVISACEnvironment(
        normalize_state=args.normalize_state,
        # 核心阈值
        comm_threshold=args.comm_threshold,
        eav_threshold=args.eav_threshold,
        # 自动校准后的参数
        comm_penalty_clip_per_user=args.comm_penalty_clip_per_user,
        comm_penalty_clip_total=args.comm_penalty_clip_total,
        comm_penalty_coef=args.comm_penalty_coef,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
        eav_penalty_coef=args.eav_penalty_coef,
        reward_scale=args.reward_scale,
        # 其他
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
    )
    
    eval_env = UAVISACEnvironment(
        normalize_state=args.normalize_state,
        comm_threshold=args.comm_threshold,
        eav_threshold=args.eav_threshold,
        comm_penalty_clip_per_user=args.comm_penalty_clip_per_user,
        comm_penalty_clip_total=args.comm_penalty_clip_total,
        comm_penalty_coef=args.comm_penalty_coef,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
        eav_penalty_coef=args.eav_penalty_coef,
        reward_scale=args.reward_scale,
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
    )
    
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(f"State size: {state_size}, Action size: {action_size}")

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

    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    print("Creating QVPO agent...")
    agent = QVPO(args, state_size, env.action_space, memory, diffusion_memory, device)

    if args.load_id is not None:
        agent.load_model(os.path.join('./results', prefix + '_' + name), id=args.load_id)

    steps = 0
    episodes = 0
    best_result = -float('inf')

    print(f"Starting training for {num_steps} steps...")
    
    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        truncated = False
        
        # 训练过程中的物理统计
        ep_comm_outage_steps = 0
        ep_secrecy_outage_steps = 0
        ep_eta_sum = 0.0
        
        state, _ = env.reset(seed=args.seed + episodes)
        episodes += 1
        
        while not (done or truncated):
            if start_steps > steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, eval=False)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            # --- 统计物理指标 (训练中) ---
            # 1. 提取或计算物理值
            p_val = (action[2] + 1)/2 * env.P_max
            eta_val, comms_val, eavs_val = calc_physics_raw(env, env.uav_position, env.user_positions, p_val)
            
            # 2. 累加统计
            ep_eta_sum += eta_val
            if np.any(comms_val < args.comm_threshold):
                ep_comm_outage_steps += 1
            if np.any(eavs_val > args.eav_threshold):
                ep_secrecy_outage_steps += 1

            # 3. 记录 Reward 组件
            if steps % 200 == 0:
                writer.add_scalar('reward_terms/eta_0', float(info.get('eta_0', 0.0)), steps)
                writer.add_scalar('reward_terms/comm_penalty_raw', float(info.get('comm_penalty', 0.0)), steps)
                writer.add_scalar('reward_terms/eav_penalty_raw', float(info.get('eav_penalty_clipped', 0.0)), steps)
                writer.add_scalar('reward_terms/reward_final', float(info.get('reward_final', reward)), steps)

            mask = 0.0 if (done or truncated) else args.gamma
            steps += 1
            episode_steps += 1
            episode_reward += reward

            agent.append_memory(state, action, reward, next_state, mask)

            if steps >= start_steps:
                agent.train(steps, updates_per_step, batch_size=batch_size, log_writer=writer)
                agent.entropy_alpha = min(args.entropy_alpha, 
                                         max(0.002, args.entropy_alpha - steps/num_steps * args.entropy_alpha))

            # --- Evaluation ---
            if steps % eval_interval == 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {steps}")
                # 传入 args 以便使用其中的 threshold
                eval_mean, comm_prob, sec_prob = evaluate(eval_env, agent, steps, source_env=env, args=args)
                
                writer.add_scalar('reward/eval_mean', eval_mean, steps)
                writer.add_scalar('Outcome/Eval_Comm_Outage_Prob', comm_prob, steps)
                writer.add_scalar('Outcome/Eval_Secrecy_Outage_Prob', sec_prob, steps)
                
                # 保存最优模型
                if eval_mean > best_result:
                    best_result = eval_mean
                    print(f"New best result: {best_result:.2f}! Saving model...")
                    agent.save_model(os.path.join('./results', prefix + '_' + name), id=id)

            state = next_state

        # --- Episode 结束后的统计 ---
        recent_rewards.append(episode_reward)
        ema_reward = episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * episode_reward)
        
        # 计算本 Episode 的平均物理表现
        ep_comm_rate = ep_comm_outage_steps / max(episode_steps, 1)
        ep_sec_rate = ep_secrecy_outage_steps / max(episode_steps, 1)
        ep_avg_eta = ep_eta_sum / max(episode_steps, 1)
        
        writer.add_scalar('reward/train', episode_reward, steps)
        writer.add_scalar('reward/train_ema', float(ema_reward), steps)
        
        # 新增：记录 Outcome 结果
        writer.add_scalar('Outcome/Train_Comm_Outage_Rate', ep_comm_rate, steps)
        writer.add_scalar('Outcome/Train_Secrecy_Outage_Rate', ep_sec_rate, steps)
        writer.add_scalar('Physics/Avg_Eta', ep_avg_eta, steps)

        print(f'Episode: {episodes:<4} Steps: {episode_steps:<4} Reward: {episode_reward:<5.1f} '
              f'CommOut: {ep_comm_rate:.0%} SecOut: {ep_sec_rate:.0%}')

        if logger is not None:
            for i in range(episode_steps):
                logger.add(epoch=steps-episode_steps+i, reward=episode_reward)

    print(f"\nTraining completed! Best result: {best_result:.2f}")
    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    args = readParser()
    if args.target_sample == -1:
        args.target_sample = args.behavior_sample

    prefix = 'qvpo'
    name = args.env_name
    keys = ("epoch", "reward")
    times = args.times
    id = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    
    result_dir = os.path.join('./results', prefix + '_' + name)
    os.makedirs(result_dir, exist_ok=True)
    
    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps)+2100, 
                   times=times, config=args, path=result_dir, id=id)

    for time in range(times):
        print(f"\n{'#'*60}\n# Starting training run {time+1}/{times}\n{'#'*60}")
        main(args, logger=logger, id=id+"_"+str(time))

    logger.save(result_dir, id=id)
    print("\nAll training runs completed!")