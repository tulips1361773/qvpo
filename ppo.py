import argparse
import os
import time
import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from myenv3 import UAVISACEnvironment
from csv_logger import CSVExperimentLogger, create_scenario_name
from logger import Logger


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
    parser = argparse.ArgumentParser(description="CleanRL-style PPO baseline for UAV-ISAC")

    parser.add_argument('--env_name', default="Env",
                        help='Custom UAV-ISAC environment (default: Env)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--num_steps', type=int, default=2500000, metavar='N',
                        help='env timesteps (default: 2500000)')
    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')
    parser.add_argument('--times', type=int, default=1, metavar='N',
                        help='times (default: 1)')

    parser.add_argument('--eav_threshold', type=float, default=10.0, metavar='G',
                        help="eavesdropper threshold in dB (default: 10.0)")
    parser.add_argument('--eav_penalty_coef', type=float, default=3.0, metavar='G',
                        help="eavesdropper penalty coefficient (default: 3.0)")
    parser.add_argument('--eav_penalty_clip_max', type=float, default=5.0, metavar='G',
                        help="max clip value for eav penalty (default: 5.0)")
    parser.add_argument('--eav_softplus_kappa', type=float, default=2.0, metavar='G',
                        help="softplus kappa for eav penalty (default: 2.0)")

    parser.add_argument('--comm_threshold', type=float, default=10.0, metavar='G',
                        help="communication threshold in dB (default: 10.0)")
    parser.add_argument('--comm_penalty_coef', type=float, default=1.5, metavar='G',
                        help="communication penalty coefficient (default: 1.5)")
    parser.add_argument('--comm_softplus_kappa', type=float, default=5.0, metavar='G',
                        help="softplus kappa for comm_penalty=softplus (default: 5.0)")
    parser.add_argument('--comm_penalty_cap_per_user', type=float, default=15.0, metavar='G',
                        help="communication penalty cap per user (default: 15.0)")
    parser.add_argument('--comm_penalty_cap_total', type=float, default=30.0, metavar='G',
                        help="communication penalty cap total (default: 30.0)")

    parser.add_argument('--action_smooth_coef', type=float, default=0.8, metavar='G',
                        help="action smoothness penalty coefficient (default: 0.8)")
    parser.add_argument('--user_move_range', type=float, default=20.0, metavar='G',
                        help="user movement range per step (default: 20.0)")
    parser.add_argument('--reward_scale', type=float, default=0.1, metavar='G',
                        help="reward scaling factor (default: 0.1)")

    if hasattr(argparse, 'BooleanOptionalAction'):
        parser.add_argument('--use_state_scaling', action=argparse.BooleanOptionalAction, default=True,
                            help="enable fixed state scaling in environment (default: True)")
    else:
        parser.add_argument('--use_state_scaling', type=_str2bool, nargs='?', const=True, default=True,
                            help="enable fixed state scaling in environment (default: True)")

    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='optimizer learning rate (default: 3e-4)')
    parser.add_argument('--rollout_steps', type=int, default=2048,
                        help='number of rollout steps per policy update (default: 2048)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='gae lambda (default: 0.95)')
    parser.add_argument('--num_minibatches', type=int, default=32,
                        help='number of minibatches per update (default: 32)')
    parser.add_argument('--update_epochs', type=int, default=10,
                        help='number of update epochs (default: 10)')
    parser.add_argument('--clip_coef', type=float, default=0.2,
                        help='PPO surrogate clipping coefficient (default: 0.2)')
    parser.add_argument('--clip_vloss', type=_str2bool, nargs='?', const=True, default=True,
                        help='use clipped value loss (default: True)')
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help='entropy coefficient (default: 0.0)')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='value function coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='gradient clipping norm (default: 0.5)')
    parser.add_argument('--target_kl', type=float, default=0.02,
                        help='target KL for early stopping (default: 0.02)')
    parser.add_argument('--anneal_lr', type=_str2bool, nargs='?', const=True, default=True,
                        help='anneal learning rate linearly (default: True)')
    parser.add_argument('--norm_adv', type=_str2bool, nargs='?', const=True, default=True,
                        help='normalize advantages (default: True)')

    parser.add_argument('--log_std_min', type=float, default=-5.0,
                        help='minimum actor log std clamp (default: -5.0)')
    parser.add_argument('--log_std_max', type=float, default=2.0,
                        help='maximum actor log std clamp (default: 2.0)')

    parser.add_argument('--eval_interval', type=int, default=10000,
                        help='evaluation interval in env steps (default: 10000)')
    parser.add_argument('--eval_episodes', type=int, default=30,
                        help='number of evaluation episodes (default: 30)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='episode log interval (default: 10)')

    return parser.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, action_space, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def _distribution(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_logstd = torch.clamp(action_logstd, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std), action_mean, action_logstd

    def get_action_and_value(self, x, action=None):
        probs, action_mean, action_logstd = self._distribution(x)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)
        return action, logprob, entropy, value

    def get_eval_action(self, x):
        action_mean = self.actor_mean(x)
        return torch.clamp(action_mean, -1.0, 1.0)


def check_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite tensor detected: {name}")


@torch.no_grad()
def evaluate(env, agent, device, steps, episodes=10):
    returns = np.zeros((episodes,), dtype=np.float32)

    eval_leakage_count = 0
    eval_total_users = 0

    legal_snr_db_list = []
    eav_snr_max_db_list = []
    eav_snr_avg_db_list = []

    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        truncated = False

        ep_legal_snr_list = []
        ep_eav_snr_max_list = []
        ep_eav_snr_avg_list = []

        while not (done or truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.get_eval_action(state_t)
            action = action.squeeze(0).cpu().numpy()
            next_state, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            state = next_state

            eval_leakage_count += info.get('leakage_count', 0)
            eval_total_users += info.get('total_users', 0)

            legal_snr = info.get('eta_0', np.nan)
            if not np.isnan(legal_snr):
                ep_legal_snr_list.append(legal_snr)

            eav_snr_list = info.get('eavesdropper_snr_list', [])
            if len(eav_snr_list) > 0:
                ep_eav_snr_max_list.append(max(eav_snr_list))
                ep_eav_snr_avg_list.append(np.mean(eav_snr_list))

        returns[i] = episode_reward
        legal_snr_db_list.append(np.mean(ep_legal_snr_list) if len(ep_legal_snr_list) > 0 else np.nan)
        eav_snr_max_db_list.append(np.mean(ep_eav_snr_max_list) if len(ep_eav_snr_max_list) > 0 else np.nan)
        eav_snr_avg_db_list.append(np.mean(ep_eav_snr_avg_list) if len(ep_eav_snr_avg_list) > 0 else np.nan)

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    eval_leakage_rate = eval_leakage_count / eval_total_users if eval_total_users > 0 else 0.0

    legal_snr_db_mean = np.nanmean(legal_snr_db_list) if len(legal_snr_db_list) > 0 else np.nan
    legal_snr_db_std = np.nanstd(legal_snr_db_list) if len(legal_snr_db_list) > 0 else np.nan
    eav_snr_max_db_mean = np.nanmean(eav_snr_max_db_list) if len(eav_snr_max_db_list) > 0 else np.nan
    eav_snr_avg_db_mean = np.nanmean(eav_snr_avg_db_list) if len(eav_snr_avg_db_list) > 0 else np.nan
    snr_gap_db_mean = legal_snr_db_mean - eav_snr_max_db_mean if (not np.isnan(legal_snr_db_mean) and not np.isnan(eav_snr_max_db_mean)) else np.nan

    print('-' * 60)
    print(f'Num steps: {steps:<7} reward: {mean_return:<8.2f} std: {std_return:<8.2f} leakage_rate: {eval_leakage_rate:.2%}')
    print(returns)
    print('-' * 60)

    return {
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


def main(args=None, logger=None, id=None):
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prefix = 'ppo'
    name = args.env_name
    run_name = id if id is not None else datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    log_dir = os.path.join('record', f'{args.env_name}', 'policy_type=PPO', f'seed={args.seed}', f'run_id={run_name}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    csv_dir = os.path.join(log_dir, 'csv_logs')
    scenario_name = create_scenario_name(args)
    csv_logger = CSVExperimentLogger(
        run_id=run_name,
        algorithm='PPO',
        seed=args.seed,
        scenario_name=scenario_name,
        eval_interval=args.eval_interval,
        csv_dir=csv_dir,
    )

    training_start_time = time.time()
    start_time = time.time()

    env = UAVISACEnvironment(
        use_state_scaling=args.use_state_scaling,
        eav_threshold=args.eav_threshold,
        eav_penalty_coef=args.eav_penalty_coef,
        eav_penalty_clip_max=args.eav_penalty_clip_max,
        eav_softplus_kappa=args.eav_softplus_kappa,
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
        eav_softplus_kappa=args.eav_softplus_kappa,
        comm_threshold=args.comm_threshold,
        comm_penalty_coef=args.comm_penalty_coef,
        comm_softplus_kappa=args.comm_softplus_kappa,
        comm_penalty_clip_per_user=args.comm_penalty_cap_per_user,
        comm_penalty_clip_total=args.comm_penalty_cap_total,
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
        reward_scale=args.reward_scale,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    print(f"State size: {obs_dim}, Action size: {action_dim}")

    agent = Agent(obs_dim, env.action_space, args.log_std_min, args.log_std_max).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.rollout_steps
    minibatch_size = batch_size // args.num_minibatches
    if batch_size % args.num_minibatches != 0:
        raise ValueError("rollout_steps must be divisible by num_minibatches")

    obs = torch.zeros((args.rollout_steps, obs_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((args.rollout_steps, action_dim), dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.rollout_steps,), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.rollout_steps,), dtype=torch.float32, device=device)
    dones = torch.zeros((args.rollout_steps,), dtype=torch.float32, device=device)
    values = torch.zeros((args.rollout_steps,), dtype=torch.float32, device=device)

    global_step = 0
    episodes = 0
    best_result = -float('inf')
    best_step = 0
    recent_rewards = deque(maxlen=100)
    recent_leakage_rates = deque(maxlen=100)
    ema_reward = None
    last_completed_episode_reward = np.nan

    train_leakage_count_global = 0
    train_total_users_global = 0
    window_leakage_count = 0
    window_total_users = 0
    ep_leakage_count = 0
    ep_total_users = 0

    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros((), dtype=torch.float32, device=device)
    episode_reward = 0.0
    episode_steps = 0

    num_updates = (args.num_steps + args.rollout_steps - 1) // args.rollout_steps

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        actual_steps = 0
        for step in range(args.rollout_steps):
            if global_step >= args.num_steps:
                break
            actual_steps += 1
            global_step += 1

            obs[step] = next_obs
            dones[step] = next_done
            check_finite_tensor("obs", next_obs)

            with torch.no_grad():
                sampled_action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                env_action = torch.clamp(sampled_action, -1.0, 1.0)
                values[step] = value.flatten()

            check_finite_tensor("sampled_action", sampled_action)
            check_finite_tensor("env_action", env_action)
            check_finite_tensor("logprob", logprob)
            check_finite_tensor("value", value)

            actions[step] = sampled_action.squeeze(0)
            logprobs[step] = logprob.squeeze(0)

            next_obs_np, reward, done, truncated, info = env.step(env_action.squeeze(0).cpu().numpy())
            if not np.isfinite(reward):
                raise RuntimeError(f"Non-finite reward detected at step {global_step}: {reward}")

            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_done = torch.tensor(float(done or truncated), dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            check_finite_tensor("next_obs", next_obs)

            step_leakage = info.get('leakage_count', 0)
            step_users = info.get('total_users', 0)
            train_leakage_count_global += step_leakage
            train_total_users_global += step_users
            window_leakage_count += step_leakage
            window_total_users += step_users
            ep_leakage_count += step_leakage
            ep_total_users += step_users

            episode_reward += reward
            episode_steps += 1

            if global_step % 200 == 0:
                writer.add_scalar('reward_terms/eta_0', float(info.get('eta_0', 0.0)), global_step)
                writer.add_scalar('reward_terms/comm_penalty', float(info.get('comm_penalty', 0.0)), global_step)
                writer.add_scalar('reward_terms/eav_penalty', float(info.get('eav_penalty', 0.0)), global_step)
                writer.add_scalar('reward_terms/eav_penalty_softplus', float(info.get('eav_penalty_softplus', 0.0)), global_step)
                writer.add_scalar('reward_terms/energy_penalty', float(info.get('energy_penalty', 0.0)), global_step)
                writer.add_scalar('reward_terms/boundary_penalty', float(info.get('boundary_penalty', 0.0)), global_step)
                writer.add_scalar('reward_terms/action_smooth_penalty', float(info.get('action_smooth_penalty', 0.0)), global_step)
                writer.add_scalar('reward_terms/reward_raw', float(info.get('reward_raw', 0.0)), global_step)
                writer.add_scalar('reward_terms/reward_clip_1', float(info.get('reward_clip_1', reward)), global_step)
                writer.add_scalar('reward_terms/reward_final', float(info.get('reward_final', reward)), global_step)
                writer.add_scalar('reward_terms/eta_0_clipped', float(info.get('eta_0_clipped', 0.0)), global_step)
                writer.add_scalar('reward_terms/comm_penalty_clipped', float(info.get('comm_penalty_clipped', 0.0)), global_step)
                writer.add_scalar('reward_terms/eav_penalty_clipped', float(info.get('eav_penalty_clipped', 0.0)), global_step)

                if step_users > 0:
                    writer.add_scalar('security/step_leakage_rate', step_leakage / step_users, global_step)
                writer.add_scalar('security/step_leakage_count', float(step_leakage), global_step)
                writer.add_scalar('security/max_eav_snr', float(info.get('max_eav_snr', 0.0)), global_step)
                writer.add_scalar('security/snr_gap_eav_raw', float(info.get('snr_gap_eav_raw', 0.0)), global_step)
                writer.add_scalar('security/eav_penalty_softplus', float(info.get('eav_penalty_softplus', 0.0)), global_step)
                writer.add_scalar('security/eav_penalty_raw', float(info.get('eav_penalty_raw', 0.0)), global_step)
                writer.add_scalar('security/eav_penalty_clipped', float(info.get('eav_penalty_clipped', 0.0)), global_step)
                writer.add_scalar('security/eav_penalty_weighted', float(info.get('eav_penalty_weighted', 0.0)), global_step)
                writer.add_scalar('security/eav_softplus_kappa', float(info.get('eav_softplus_kappa', 0.0)), global_step)

                if window_total_users > 0:
                    writer.add_scalar('security/train_leakage_rate_window200', window_leakage_count / window_total_users, global_step)
                else:
                    writer.add_scalar('security/train_leakage_rate_window200', 0.0, global_step)
                window_leakage_count = 0
                window_total_users = 0

                if train_total_users_global > 0:
                    writer.add_scalar('security/train_leakage_rate_global', train_leakage_count_global / train_total_users_global, global_step)

            if done or truncated:
                episodes += 1
                last_completed_episode_reward = episode_reward
                recent_rewards.append(episode_reward)
                ema_reward = episode_reward if ema_reward is None else (0.95 * ema_reward + 0.05 * episode_reward)

                if ep_total_users > 0:
                    recent_leakage_rates.append(ep_leakage_count / ep_total_users)
                ep_leakage_count = 0
                ep_total_users = 0

                writer.add_scalar('reward/train', episode_reward, global_step)
                writer.add_scalar('reward/train_ma100', float(np.mean(recent_rewards)), global_step)
                writer.add_scalar('reward/train_ema', float(ema_reward), global_step)
                if len(recent_leakage_rates) > 0:
                    writer.add_scalar('security/train_leakage_rate_ma100', float(np.mean(recent_leakage_rates)), global_step)

                if episodes % args.log_interval == 0:
                    print(f'Episode: {episodes:<4}  Steps: {episode_steps:<4}  Total Steps: {global_step:<7}  Reward: {episode_reward:<7.2f}')

                if logger is not None:
                    for i in range(episode_steps):
                        logger.add(epoch=global_step - episode_steps + i, reward=episode_reward)

                next_obs_np, _ = env.reset(seed=args.seed + episodes)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.zeros((), dtype=torch.float32, device=device)
                episode_reward = 0.0
                episode_steps = 0

            if global_step % args.eval_interval == 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {global_step}")
                print(f"{'='*60}")
                eval_results = evaluate(eval_env, agent, device, global_step, episodes=args.eval_episodes)
                writer.add_scalar('reward/eval_mean', eval_results['mean_return'], global_step)
                writer.add_scalar('security/eval_leakage_rate', eval_results['eval_leakage_rate'], global_step)

                time_elapsed = time.time() - training_start_time
                train_reward_ma = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else np.nan
                csv_logger.log_training_metrics(
                    eval_results=eval_results,
                    step=global_step,
                    time_elapsed_sec=time_elapsed,
                    train_reward=last_completed_episode_reward,
                    train_reward_ma100=train_reward_ma
                )

                if eval_results['mean_return'] > best_result:
                    best_result = eval_results['mean_return']
                    best_step = global_step
                    print(f"New best result: {best_result:.2f}! Saving model...")
                    result_dir = os.path.join('./results', f'{prefix}_{name}')
                    os.makedirs(result_dir, exist_ok=True)
                    torch.save(agent.state_dict(), os.path.join(result_dir, f'{run_name}_best.pt'))

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1)
            advantages = torch.zeros((actual_steps,), dtype=torch.float32, device=device)
            returns = torch.zeros((actual_steps,), dtype=torch.float32, device=device)
            lastgaelam = 0.0
            for t in reversed(range(actual_steps)):
                if t == actual_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.squeeze(0)
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values[:actual_steps]

        b_obs = obs[:actual_steps].reshape((-1, obs_dim))
        b_logprobs = logprobs[:actual_steps].reshape(-1)
        b_actions = actions[:actual_steps].reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values[:actual_steps].reshape(-1)

        check_finite_tensor("b_advantages", b_advantages)
        check_finite_tensor("b_returns", b_returns)
        check_finite_tensor("b_values", b_values)

        b_inds = np.arange(actual_steps)
        clipfracs = []
        old_approx_kl = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)

        minibatch_size_eff = max(1, actual_steps // args.num_minibatches)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, actual_steps, minibatch_size_eff):
                end = min(start + minibatch_size_eff, actual_steps)
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                logratio = torch.clamp(logratio, -20.0, 20.0)
                ratio = torch.exp(logratio)

                check_finite_tensor("newlogprob", newlogprob)
                check_finite_tensor("ratio", ratio)
                check_finite_tensor("newvalue", newvalue)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv and mb_advantages.numel() > 1:
                    adv_std = mb_advantages.std(unbiased=False)
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (adv_std + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                check_finite_tensor("loss", loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("ppo/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("ppo/value_loss", v_loss.item(), global_step)
        writer.add_scalar("ppo/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("ppo/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("ppo/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("ppo/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("ppo/clipfrac", np.mean(clipfracs) if len(clipfracs) > 0 else 0.0, global_step)
        writer.add_scalar("ppo/explained_variance", explained_var, global_step)
        writer.add_scalar("ppo/sps", int(global_step / max(1e-6, (time.time() - start_time))), global_step)

        if global_step >= args.num_steps:
            break

    print(f"\n{'='*60}")
    print("Training completed! Performing final evaluation...")
    print(f"{'='*60}")
    final_eval_results = evaluate(eval_env, agent, device, global_step, episodes=args.eval_episodes)
    training_total_time = time.time() - training_start_time

    csv_logger.log_final_comparison(
        final_eval_results=final_eval_results,
        total_train_steps=global_step,
        best_eval_reward=best_result,
        best_step=best_step,
        training_time_sec=training_total_time
    )

    print(f"Total episodes: {episodes}")
    print(f"Total steps: {global_step}")
    print(f"Best evaluation result: {best_result:.2f} at step {best_step}")
    print(f"Final evaluation result: {final_eval_results['mean_return']:.2f}")
    print(f"Final leakage rate: {final_eval_results['eval_leakage_rate']:.2%}")
    print(f"Training time: {training_total_time:.2f} seconds")
    print(f"CSV logs saved to: {csv_dir}")
    print(f"{'='*60}")

    result_dir = os.path.join('./results', f'{prefix}_{name}')
    os.makedirs(result_dir, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(result_dir, f'{run_name}_final.pt'))

    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    args = readParser()

    prefix = 'ppo'
    name = args.env_name
    keys = ("epoch", "reward")
    times = args.times
    id = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    print(f"\n{'#'*60}")
    print(f"# PPO Training Configuration")
    print(f"# Seed: {args.seed}")
    print(f"# Total steps: {args.num_steps}")
    print(f"# Run ID: {id}")
    print(f"# use_state_scaling: {args.use_state_scaling}")
    print(f"{'#'*60}\n")

    result_dir = os.path.join('./results', prefix + '_' + name)
    os.makedirs(result_dir, exist_ok=True)

    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps) + 2100,
                    times=times, config=args, path=result_dir, id=id)

    for run_idx in range(times):
        print(f"\n{'#'*60}")
        print(f"# Starting training run {run_idx+1}/{times}")
        print(f"{'#'*60}\n")
        main(args, logger=logger, id=id + "_" + str(run_idx))

    logger.save(result_dir, id=id)
    print("\nAll training runs completed!")
