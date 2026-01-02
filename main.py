import argparse
import copy

import numpy as np
import torch

from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory, DiffusionMemory

from tensorboardX import SummaryWriter
import gymnasium as gym  # 修改1: 使用gymnasium替代gym
import os
from logger import Logger
import datetime

# 修改2: 导入自定义环境
from myenv import UAVISACEnvironment

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy for UAV-ISAC')
    
    # 修改3: 环境名称改为自定义环境
    parser.add_argument('--env_name', default="Env",
                        help='Custom UAV-ISAC environment (default: Env)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    # 修改4: 根据UAV环境特性调整训练步数
    parser.add_argument('--num_steps', type=int, default=2500000, metavar='N',
                        help='env timesteps (default: 2500000, 增加以适应复杂环境)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    
    # 修改5: 调整折扣因子（UAV环境episode较短，可用更高的gamma）
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=20, metavar='N',
                        help='diffusion timesteps (default: 20)')
    
    # 修改6: 调整学习率（可能需要更小的学习率以应对高维状态）
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

    # 修改7: 动作空间较小(3维)，可适当减少采样数
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
    
    # 修改8: 增加熵系数以鼓励探索（UAV环境可能需要更多探索）
    parser.add_argument('--entropy_alpha', type=float, default=0.05, metavar='G', 
                        help="entropy_alpha (default: 0.05, 增加探索)")

    return parser.parse_args()


def evaluate(env, agent, steps):
    """评估函数 - 修改9: 适配gymnasium的返回格式"""
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        # 修改10: gymnasium的reset返回(obs, info)
        state, _ = env.reset()
        episode_reward = 0.
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.sample_action(state, eval=True)
            # 修改11: gymnasium的step返回5个值
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        returns[i] = episode_reward

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'reward: {mean_return:<5.1f}  '
          f'std: {std_return:<5.1f}')
    print(returns)
    print('-' * 60)
    return mean_return


def main(args=None, logger=None, id=None):

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir = "record"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', 
                          f'ratio={args.ratio}', f'seed={args.seed}')
    writer = SummaryWriter(log_dir)

    # 修改12: 创建自定义环境
    print("Initializing UAV-ISAC Environment...")
    env = gym.make(args.env_name)
    
    # 修改13: gymnasium环境的deepcopy可能有问题，建议重新创建
    eval_env = gym.make(args.env_name)
    
    # 获取状态和动作维度
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 修改14: gymnasium使用reset(seed=...)而不是env.seed()
    # 注意：seed在reset时设置，这里先不调用

    # 训练参数
    memory_size = 1e6
    num_steps = args.num_steps
    
    # 修改15: 增加初始探索步数（复杂环境需要更多探索）
    start_steps = 25000  # 从10000增加到25000
    eval_interval = 10000
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10

    # 创建经验池
    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    # 创建QVPO智能体
    print("Creating QVPO agent...")
    agent = QVPO(args, state_size, env.action_space, memory, diffusion_memory, device)

    steps = 0
    episodes = 0
    best_result = -float('inf')

    print(f"Starting training for {num_steps} steps...")
    print(f"Random exploration for first {start_steps} steps")

    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        truncated = False
        
        # 修改16: gymnasium的reset返回(obs, info)
        state, _ = env.reset(seed=args.seed + episodes)  # 每个episode使用不同seed
        episodes += 1
        
        while not (done or truncated):
            # 动作选择
            if start_steps > steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, eval=False)
            
            # 修改17: gymnasium的step返回5个值
            next_state, reward, done, truncated, info = env.step(action)

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
                tmp_result = evaluate(eval_env, agent, steps)
                
                if tmp_result > best_result:
                    best_result = tmp_result
                    print(f"New best result: {best_result:.2f}! Saving model...")
                    agent.save_model(os.path.join('./results', prefix + '_' + name), id=id)

            state = next_state

        # Episode结束后的日志
        if episodes % log_interval == 0:
            writer.add_scalar('reward/train', episode_reward, steps)

        print(f'Episode: {episodes:<4}  '
              f'Steps: {episode_steps:<4}  '
              f'Total Steps: {steps:<7}  '
              f'Reward: {episode_reward:<5.1f}')

        if logger is not None:
            for i in range(episode_steps):
                logger.add(epoch=steps-episode_steps+i, reward=episode_reward)

    # 训练结束
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total episodes: {episodes}")
    print(f"Best evaluation result: {best_result:.2f}")
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
    
    # 创建结果目录
    result_dir = os.path.join('./results', prefix + '_' + name)
    os.makedirs(result_dir, exist_ok=True)
    
    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps)+2100, 
                   times=times, config=args, path=result_dir, id=id)

    ## 运行训练
    for time in range(times):
        print(f"\n{'#'*60}")
        print(f"# Starting training run {time+1}/{times}")
        print(f"{'#'*60}\n")
        main(args, logger=logger, id=id+"_"+str(time))

    logger.save(result_dir, id=id)
    print("\nAll training runs completed!")