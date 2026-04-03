import argparse
import copy
import datetime
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from csv_logger import CSVExperimentLogger, create_scenario_name
from myenv3 import UAVISACEnvironment

# ============================================================
# Utilities
# ============================================================

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


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.not_dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, next_state, not_done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.not_dones[self.ptr] = not_done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.states[idxs], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], dtype=torch.float32, device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, not_dones


# ============================================================
# TD3 Networks
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 1)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_fc3(q1)


class TD3Agent:
    def __init__(self, args, state_dim, action_space, device):
        self.device = device
        self.action_dim = int(np.prod(action_space.shape))
        self.action_low = action_space.low.astype(np.float32)
        self.action_high = action_space.high.astype(np.float32)

        self.actor = Actor(state_dim, self.action_dim, self.action_low, self.action_high).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq

        self.exploration_noise_start = args.exploration_noise
        self.exploration_noise_min = args.exploration_noise_min
        self.exploration_noise_anneal_end = args.exploration_noise_anneal_end
        self.exploration_noise_anneal_start = args.exploration_noise_anneal_start
        self.total_updates = 0

    def get_exploration_noise(self, step, start_steps):
        anneal_start = self.exploration_noise_anneal_start if self.exploration_noise_anneal_start >= 0 else start_steps
        if step <= anneal_start:
            return self.exploration_noise_start
        if step >= self.exploration_noise_anneal_end:
            return self.exploration_noise_min
        ratio = (step - anneal_start) / float(max(1, self.exploration_noise_anneal_end - anneal_start))
        return self.exploration_noise_start + ratio * (self.exploration_noise_min - self.exploration_noise_start)

    @torch.no_grad()
    def sample_action(self, state, eval=False, step=None, start_steps=0):
        state_t = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        action = self.actor(state_t).cpu().numpy().flatten()
        if not eval:
            cur_noise_std = self.get_exploration_noise(step if step is not None else 0, start_steps)
            noise = np.random.normal(0.0, cur_noise_std, size=action.shape).astype(np.float32)
            action = action + noise
        action = np.clip(action, self.action_low, self.action_high)
        return action

    def train(self, replay_buffer, batch_size=256, log_writer=None, t=None):
        self.total_updates += 1
        states, actions, rewards, next_states, not_dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            action_high = torch.as_tensor(self.action_high, dtype=torch.float32, device=self.device)
            action_low = torch.as_tensor(self.action_low, dtype=torch.float32, device=self.device)
            next_actions = (self.actor_target(next_states) + noise).clamp(action_low, action_high)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + not_dones * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = 0.5 * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss_value = None
        if self.total_updates % self.policy_freq == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.item()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if log_writer is not None and t is not None and self.total_updates % 200 == 0:
            log_writer.add_scalar('loss/critic', critic_loss.item(), t)
            log_writer.add_scalar('q/current_q1_mean', current_q1.mean().item(), t)
            log_writer.add_scalar('q/current_q2_mean', current_q2.mean().item(), t)
            log_writer.add_scalar('q/target_q_mean', target_q.mean().item(), t)
            log_writer.add_scalar('q/reward_mean', rewards.mean().item(), t)
            if actor_loss_value is not None:
                log_writer.add_scalar('loss/actor', actor_loss_value, t)

    def save_model(self, dir_path, id=None):
        os.makedirs(dir_path, exist_ok=True)
        suffix = f'_{id}' if id is not None else ''
        torch.save(self.actor.state_dict(), os.path.join(dir_path, f'actor{suffix}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(dir_path, f'critic{suffix}.pth'))

    def load_model(self, dir_path, id=None):
        suffix = f'_{id}' if id is not None else ''
        self.actor.load_state_dict(torch.load(os.path.join(dir_path, f'actor{suffix}.pth'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(dir_path, f'critic{suffix}.pth'), map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


# ============================================================
# Evaluation & Args
# ============================================================

def evaluate(env, agent, steps, episodes=30):
    returns = np.zeros((episodes,), dtype=np.float32)
    eval_leakage_count, eval_total_users = 0, 0
    legal_snr_db_list, eav_snr_max_db_list, eav_snr_avg_db_list = [], [], []

    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = truncated = False
        ep_legal_snr_list, ep_eav_snr_max_list, ep_eav_snr_avg_list = [], [], []

        while not (done or truncated):
            action = agent.sample_action(state, eval=True)
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

        if len(ep_legal_snr_list) > 0:
            legal_snr_db_list.append(np.mean(ep_legal_snr_list))
        else:
            legal_snr_db_list.append(np.nan)

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
    eval_leakage_rate = eval_leakage_count / eval_total_users if eval_total_users > 0 else 0.0
    legal_snr_db_mean = np.nanmean(legal_snr_db_list) if legal_snr_db_list else np.nan
    legal_snr_db_std = np.nanstd(legal_snr_db_list) if legal_snr_db_list else np.nan
    eav_snr_max_db_mean = np.nanmean(eav_snr_max_db_list) if eav_snr_max_db_list else np.nan
    eav_snr_avg_db_mean = np.nanmean(eav_snr_avg_db_list) if eav_snr_avg_db_list else np.nan

    if not np.isnan(legal_snr_db_mean) and not np.isnan(eav_snr_max_db_mean):
        snr_gap_db_mean = legal_snr_db_mean - eav_snr_max_db_mean
    else:
        snr_gap_db_mean = np.nan

    print('-' * 60)
    print(f'Steps: {steps:<7}  reward: {mean_return:<7.2f}  std: {std_return:<7.2f}  leakage_rate: {eval_leakage_rate:.2%}')
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


def readParser():
    parser = argparse.ArgumentParser(description='Optimized TD3 for UAV-ISAC')
    parser.add_argument('--env_name', default='Env')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=2500000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--cuda', default='cuda:0')
    parser.add_argument('--load_id', type=str, default=None)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--policy_type', type=str, default='TD3')
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--exploration_noise_min', type=float, default=0.03)
    parser.add_argument('--exploration_noise_anneal_end', type=int, default=600000)
    parser.add_argument('--exploration_noise_anneal_start', type=int, default=-1)

    # 环境参数
    parser.add_argument('--eav_threshold', type=float, default=10.0)
    parser.add_argument('--eav_penalty_coef', type=float, default=3.0)
    parser.add_argument('--eav_penalty_cap', type=float, default=20.0)
    parser.add_argument('--comm_threshold', type=float, default=10.0)
    parser.add_argument('--comm_penalty_coef', type=float, default=1)
    parser.add_argument('--comm_softplus_kappa', type=float, default=5.0)
    parser.add_argument('--comm_penalty_cap_per_user', type=float, default=15.0)
    parser.add_argument('--comm_penalty_cap_total', type=float, default=30.0)
    parser.add_argument('--comm_penalty_avg_over_k', type=_str2bool, default=True)
    parser.add_argument('--action_smooth_coef', type=float, default=0.1)
    parser.add_argument('--user_move_range', type=float, default=20.0)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    parser.add_argument('--eta_clip_max', type=float, default=15.0)
    parser.add_argument('--eav_penalty_clip_max', type=float, default=5.0)
    parser.add_argument('--eav_softplus_kappa', type=float, default=2.0)
    parser.add_argument('--use_state_scaling', type=_str2bool, default=True)
    parser.add_argument('--use_obs_normalizer', type=_str2bool, default=False)

    # 与主算法环境接口对齐，仅补参数，不改 TD3 逻辑
    parser.add_argument('--E_tot', type=int, default=25000)

    return parser.parse_args()


def build_env(args):
    return UAVISACEnvironment(
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
        E_tot=args.E_tot,
    )


def main(args=None, id=None):
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

    log_dir = os.path.join('record', args.env_name, args.policy_type, f'ratio={args.ratio}', f'seed={args.seed}')
    if id is not None:
        log_dir = os.path.join(log_dir, f'run_id={id}')
    writer = SummaryWriter(log_dir)

    csv_dir = os.path.join(log_dir, 'csv_logs')
    csv_logger = CSVExperimentLogger(
        run_id=id or '0',
        algorithm='TD3',
        seed=args.seed,
        scenario_name=create_scenario_name(args),
        eval_interval=10000,
        csv_dir=csv_dir
    )

    training_start_time = time.time()

    env, eval_env = build_env(args), build_env(args)
    state_size = int(np.prod(env.observation_space.shape))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory = ReplayBuffer(state_size, int(np.prod(env.action_space.shape)), 1000000, device)
    agent = TD3Agent(args, state_size, env.action_space, device)

    if args.load_id:
        agent.load_model('./results/td3_Env', id=args.load_id)

    steps, episodes = 0, 0
    best_result = -float('inf')
    best_step = 0
    last_ep_reward = np.nan
    recent_rewards = deque(maxlen=100)

    while steps < args.num_steps:
        state, _ = env.reset(seed=args.seed + episodes)
        episodes += 1
        ep_reward, ep_steps = 0.0, 0
        done = truncated = False

        while not (done or truncated):
            if steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, step=steps, start_steps=args.start_steps)

            next_state, reward, done, truncated, info = env.step(action)
            memory.append(state, action, reward, next_state, 0.0 if (done or truncated) else 1.0)

            steps += 1
            ep_steps += 1
            ep_reward += reward

            if steps >= args.start_steps:
                agent.train(memory, args.batch_size, writer, steps)

            if steps % 200 == 0:
                writer.add_scalar('reward_terms/reward_final', float(reward), steps)
                if steps >= args.start_steps:
                    writer.add_scalar(
                        'policy/exploration_noise_std',
                        agent.get_exploration_noise(steps, args.start_steps),
                        steps
                    )

            if steps % 10000 == 0:
                eval_res = evaluate(eval_env, agent, steps)
                writer.add_scalar('reward/eval_mean', eval_res['mean_return'], steps)

                time_elapsed = time.time() - training_start_time
                csv_logger.log_training_metrics(
                    eval_results=eval_res,
                    step=steps,
                    time_elapsed_sec=time_elapsed,
                    train_reward=last_ep_reward,
                    train_reward_ma100=np.mean(recent_rewards) if recent_rewards else np.nan
                )

                if eval_res['mean_return'] > best_result:
                    best_result = eval_res['mean_return']
                    best_step = steps
                    agent.save_model(os.path.join('./results', 'td3_' + args.env_name), id=id)

            state = next_state
            if steps >= args.num_steps:
                break

        last_ep_reward = ep_reward
        recent_rewards.append(ep_reward)
        writer.add_scalar('reward/train', ep_reward, steps)
        print(f'Ep: {episodes} | Steps: {steps} | Reward: {ep_reward:.2f}')

    # ============================================================
    # 训练结束后的最终独立评估 + final CSV
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Training completed! Performing final evaluation...")
    print(f"{'=' * 60}")

    final_eval_results = evaluate(eval_env, agent, steps, episodes=30)
    training_total_time = time.time() - training_start_time

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
    print(f"{'=' * 60}")

    env.close()
    eval_env.close()
    writer.close()


if __name__ == '__main__':
    args = readParser()
    main(args, id=datetime.datetime.now().strftime('%y%m%d_%H%M%S'))