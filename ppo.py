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
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# 导入自定义环境
from myenv import UAVISACEnvironment

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Agent for UAV-ISAC')
    
    # --- PPO 核心参数 (已修正默认值) ---
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    # 🔥 关键修正：总步数由 total_timesteps 控制
    parser.add_argument('--total_timesteps', type=int, default=2500000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    # 修正 cuda 参数为字符串，兼容 main.py
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='device to use: cuda:0, cuda:1 or cpu')
    
    # 🔥 关键修正：num_steps 是每次更新采集的步数，必须小！通常 2048
    parser.add_argument('--num_steps', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--num_minibatches', type=int, default=32,
                        help='the number of mini-batches')
    parser.add_argument('--update_epochs', type=int, default=10,
                        help='the K epochs to update the policy')
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True,
                        help="Toggle learning rate annealing")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--clip_coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--clip_vloss', type=lambda x: bool(strtobool(x)), default=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument('--ent_coef', type=float, default=0.01, # 稍微增加熵系数防止过早收敛
                        help="coefficient of the entropy")
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument('--target_kl', type=float, default=None,
                        help='the target KL divergence threshold')
    
    # --- 环境参数 (保持一致) ---
    parser.add_argument('--env_name', default="Env", help='Custom UAV-ISAC environment')
    parser.add_argument('--normalize_state', type=lambda x: bool(strtobool(x)), default=True,
                        help="enable state normalization (default: True)")
    
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
    parser.add_argument('--comm_penalty_avg_over_k', type=lambda x: bool(strtobool(x)), default=True)
    
    parser.add_argument('--action_smooth_coef', type=float, default=0.8)
    parser.add_argument('--user_move_range', type=float, default=20.0)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    
    parser.add_argument('--eta_clip_max', type=float, default=15.0)
    parser.add_argument('--comm_penalty_clip_max', type=float, default=5.0)
    parser.add_argument('--eav_penalty_clip_max', type=float, default=5.0)

    args = parser.parse_args()
    args.batch_size = int(args.num_steps) 
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.prod(env.observation_space.shape)
        action_shape = np.prod(env.action_space.shape)
        
        # 🔥 修改：增加网络宽度至 256 (对齐 SAC)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_shape), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_eval_action(self, x):
        """Evaluation模式下使用均值（确定性策略）"""
        return self.actor_mean(x)

def evaluate(env, agent, device, source_env=None):
    """PPO 评估函数"""
    if source_env is not None and hasattr(source_env, 'state_normalizer') and hasattr(env, 'state_normalizer'):
        env.state_normalizer.mean = source_env.state_normalizer.mean.copy()
        env.state_normalizer.var = source_env.state_normalizer.var.copy()
        env.state_normalizer.count = source_env.state_normalizer.count
    
    if hasattr(env, 'state_normalizer'):
        env.state_normalizer.set_training(False)
    
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)
    
    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.
        done = False
        truncated = False
        
        while not (done or truncated):
            state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.get_eval_action(state_tensor)
                action = action.cpu().numpy()[0]
            
            # Action clip
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        returns[i] = episode_reward
    
    if hasattr(env, 'state_normalizer'):
        env.state_normalizer.set_training(True)
    
    mean_return = np.mean(returns)
    return mean_return

def main():
    args = parse_args()
    
    # 构造运行名称，包含主要参数
    run_name = f"{args.env_name}__ppo_s{args.seed}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join("record", "PPO_Fixed", run_name)
    writer = SummaryWriter(log_dir)
    
    # 打印参数
    print(f"{'='*30}")
    print(f"Running PPO with:")
    print(f"  Total Timesteps: {args.total_timesteps}")
    print(f"  Steps per Rollout: {args.num_steps} (Updates every {args.num_steps} steps)")
    print(f"  Hidden Size: 256")
    print(f"  Device: {args.cuda}")
    print(f"{'='*30}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 设备选择修复
    if torch.cuda.is_available() and "cuda" in args.cuda:
        device = torch.device(args.cuda)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 环境参数
    env_kwargs = {
        'normalize_state': args.normalize_state,
        'eav_agg': args.eav_agg,
        'eav_logsumexp_kappa': args.eav_logsumexp_kappa,
        'eav_threshold': args.eav_threshold,
        'eav_penalty_coef': args.eav_penalty_coef,
        'eav_penalty_cap': args.eav_penalty_cap,
        'comm_penalty_type': args.comm_penalty,
        'comm_threshold': args.comm_threshold,
        'comm_penalty_coef': args.comm_penalty_coef,
        'comm_softplus_kappa': args.comm_softplus_kappa,
        'comm_huber_delta': args.comm_huber_delta,
        'comm_penalty_cap_per_user': args.comm_penalty_cap_per_user,
        'comm_penalty_cap_total': args.comm_penalty_cap_total,
        'comm_penalty_avg_over_k': args.comm_penalty_avg_over_k,
        'action_smooth_coef': args.action_smooth_coef,
        'user_move_range': args.user_move_range,
        'reward_scale': args.reward_scale,
        'eta_clip_max': args.eta_clip_max,
        'comm_penalty_clip_max': args.comm_penalty_clip_max,
        'eav_penalty_clip_max': args.eav_penalty_clip_max,
    }

    env = UAVISACEnvironment(**env_kwargs)
    eval_env = UAVISACEnvironment(**env_kwargs)

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 存储Buffer
    obs = torch.zeros((args.num_steps, int(np.prod(env.observation_space.shape)))).to(device)
    actions = torch.zeros((args.num_steps, int(np.prod(env.action_space.shape)))).to(device)
    logprobs = torch.zeros((args.num_steps)).to(device)
    rewards = torch.zeros((args.num_steps)).to(device)
    dones = torch.zeros((args.num_steps)).to(device)
    values = torch.zeros((args.num_steps)).to(device)

    global_step = 0
    start_time = time.time()
    
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    
    episode_rewards = deque(maxlen=100)
    current_episode_reward = 0
    
    # Eval 频率控制
    last_eval_step = 0
    eval_interval = 10000 # 每 10000 步评估一次
    best_eval_return = -float('inf')

    info_buffer = {k: [] for k in ['eta_0', 'comm_penalty', 'eav_penalty', 'reward_raw', 'reward_final']}

    num_updates = args.total_timesteps // args.num_steps

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- 1. Rollout ---
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.tensor(float(done)).to(device)

            current_episode_reward += reward

            # Buffer info
            for k in info_buffer.keys():
                if k in info:
                    info_buffer[k].append(info[k])

            if done:
                episode_rewards.append(current_episode_reward)
                writer.add_scalar("reward/train", current_episode_reward, global_step)
                writer.add_scalar("reward/train_ma100", np.mean(episode_rewards), global_step)
                
                current_episode_reward = 0
                next_obs, _ = env.reset(seed=args.seed + global_step)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(1).to(device)

        # 记录分项
        for k, v in info_buffer.items():
            if v: writer.add_scalar(f"reward_terms/{k}", np.mean(v), global_step)
        for k in info_buffer.keys(): info_buffer[k] = []

        # --- 2. GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flattening the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # Normalize Advantage (关键：有助于稳定训练)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 记录训练曲线
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # ============================================================
        # 修正后的评估逻辑 (基于 global_step)
        # ============================================================
        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Step {global_step}: Evaluating...")
            
            # 评估
            eval_return = evaluate(eval_env, agent, device, source_env=env)
            
            # 记录 Eval Reward
            writer.add_scalar("reward/eval_mean", eval_return, global_step)
            print(f"Eval Reward: {eval_return:.2f} (Best: {best_eval_return:.2f})")

            # 保存最优模型
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                model_path = os.path.join(log_dir, "agent_best.pt")
                torch.save(agent.state_dict(), model_path)
                print(f"New best model saved to {model_path}")
                
                # 保存归一化统计量
                if args.normalize_state:
                    import pickle
                    with open(os.path.join(log_dir, "obs_rms.pkl"), 'wb') as f:
                        pickle.dump({
                            'mean': env.state_normalizer.mean,
                            'var': env.state_normalizer.var,
                            'count': env.state_normalizer.count
                        }, f)

    env.close()
    eval_env.close()
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best evaluation result: {best_eval_return:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()