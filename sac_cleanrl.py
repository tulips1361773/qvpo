# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

# 🔥 导入你的自定义环境
from myenv import UAVISACEnvironment

def parse_args():
    parser = argparse.ArgumentParser()
    # ==================== 实验基础设置 ====================
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True,
    #     help="if toggled, cuda will be enabled by default")
    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="uav-isac-sac",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    # ==================== SAC 算法超参数 ====================
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denoted as d in sutton book
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True,
        help="automatic tuning of the entropy coefficient")

    # ==================== 你的 UAV 环境参数 (与 main.py 保持一致) ====================
    parser.add_argument('--normalize_state', type=lambda x: bool(strtobool(str(x))), default=True)
    parser.add_argument('--eav_agg', type=str, default='top2')
    parser.add_argument('--eav_logsumexp_kappa', type=float, default=0.5)
    parser.add_argument('--eav_threshold', type=float, default=10.0)
    parser.add_argument('--eav_penalty_coef', type=float, default=0.5)
    parser.add_argument('--eav_penalty_cap', type=float, default=20.0)
    parser.add_argument('--comm_penalty', type=str, default='softplus')
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

    args = parser.parse_args()
    return args

# 辅助函数：实例化你的环境
def make_uav_env(args):
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
    return env

# ALGO LOGIC: initialize Agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # 你的 obs_dim 和 action_dim
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
        # Action rescaling is handled by Tanh, range is [-1, 1] matching your Env

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        # SAC paper tricks: log_std bounds
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t # Range [-1, 1]
        
        # Calculate log_prob
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (correction formula for tanh)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # mean is used for deterministic evaluation
        mean = torch.tanh(mean)
        return action, log_prob, mean

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Setup logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Running on device: {device}")

    # Env setup
    # 我们实例化两个环境：一个用于训练，一个用于评估
    # 注意：你的环境不使用 gym.vector，因为 state_normalizer 在内部维护
    env = make_uav_env(args)
    eval_env = make_uav_env(args)

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # Agent setup
    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Replay buffer (Simple numpy implementation)
    # 因为没有使用 gym vector，obs shape 不需要处理
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    rb_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_actions = np.zeros((args.buffer_size, *act_shape), dtype=np.float32)
    rb_rewards = np.zeros((args.buffer_size), dtype=np.float32)
    rb_next_obs = np.zeros((args.buffer_size, *obs_shape), dtype=np.float32)
    rb_dones = np.zeros((args.buffer_size), dtype=np.float32)
    
    rb_state = {"ptr": 0, "size": 0}

    def add_to_buffer(obs, act, rew, next_obs, done):
        ptr = rb_state["ptr"]
        rb_obs[ptr] = obs
        rb_actions[ptr] = act
        rb_rewards[ptr] = rew
        rb_next_obs[ptr] = next_obs
        rb_dones[ptr] = done
        rb_state["ptr"] = (ptr + 1) % args.buffer_size
        rb_state["size"] = min(rb_state["size"] + 1, args.buffer_size)

    def sample_buffer(batch_size):
        size = rb_state["size"]
        idxs = np.random.randint(0, size, size=batch_size)
        return (
            torch.tensor(rb_obs[idxs], device=device),
            torch.tensor(rb_actions[idxs], device=device),
            torch.tensor(rb_rewards[idxs], device=device),
            torch.tensor(rb_next_obs[idxs], device=device),
            torch.tensor(rb_dones[idxs], device=device),
        )

    # Training Loop
    start_time = time.time()
    obs, _ = env.reset(seed=args.seed)
    
    # For logging
    episode_reward = 0
    episode_length = 0

    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _, _ = actor.get_action(torch.Tensor(obs).to(device).unsqueeze(0))
                action = action.cpu().numpy().flatten() # [Batch, Action] -> [Action]

        # Step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Log specific UAV metrics periodically
        if global_step % 1000 == 0:
            writer.add_scalar("charts/eta_0", info.get('eta_0', 0), global_step)
            writer.add_scalar("charts/comm_penalty", info.get('comm_penalty', 0), global_step)
            writer.add_scalar("charts/eav_penalty", info.get('eav_penalty', 0), global_step)
            writer.add_scalar("charts/energy_penalty", info.get('energy_penalty', 0), global_step)

        episode_reward += reward
        episode_length += 1
        
        # Handle done/truncated for buffer
        real_done = done # In standard SAC, we don't treat timeout as done for value update usually, but simple done is fine here
        
        add_to_buffer(obs, action, reward, next_obs, real_done)
        
        obs = next_obs

        if done or truncated:
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            print(f"Global Step: {global_step}, Ep Return: {episode_reward:.2f}")
            obs, _ = env.reset(seed=args.seed)
            episode_reward = 0
            episode_length = 0

        # Optimization step
        if global_step > args.learning_starts:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = sample_buffer(args.batch_size)

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

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
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

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        # Periodic Evaluation (Crucial for your paper)
        if global_step > 0 and global_step % 5000 == 0:
            # 🔥 关键：同步归一化参数！
            if args.normalize_state:
                eval_env.state_normalizer.mean = env.state_normalizer.mean.copy()
                eval_env.state_normalizer.var = env.state_normalizer.var.copy()
                eval_env.state_normalizer.count = env.state_normalizer.count
                eval_env.state_normalizer.set_training(False) # 关闭评估环境的更新
            
            avg_ret = 0
            eval_episodes = 10
            for _ in range(eval_episodes):
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_ret = 0
                while not eval_done:
                    with torch.no_grad():
                        # 使用 mean (deterministic) 进行评估
                        _, _, eval_action = actor.get_action(torch.Tensor(eval_obs).to(device).unsqueeze(0))
                        eval_action = eval_action.cpu().numpy().flatten()
                    eval_obs, r, eval_done, _, _ = eval_env.step(eval_action)
                    eval_ret += r
                avg_ret += eval_ret
            avg_ret /= eval_episodes
            
            writer.add_scalar("eval/return", avg_ret, global_step)
            print(f"Eval at step {global_step}: {avg_ret:.2f}")

            # 恢复
            if args.normalize_state:
                eval_env.state_normalizer.set_training(True)

    env.close()
    eval_env.close()
    writer.close()