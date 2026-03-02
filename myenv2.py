import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from gymnasium.envs.registration import register

# 注册环境
try:
    register(
        id='Env-v0',
        entry_point='myenv2:UAVISACEnvironment', # 注意文件名匹配
        max_episode_steps=50
    )
except:
    pass

def calc_energy(v_u_t: float, delta_t: float) -> float:
    """根据论文公式(13) 计算单个时隙能耗 (J)。"""
    d_0   = 0.6
    rho_a = 1.225
    z     = 0.05
    G     = 0.503
    P_s   = 79.85
    U_r   = 120.0
    P_m   = 88.63
    V_h   = 4.03

    term1 = 0.5 * d_0 * rho_a * z * G * v_u_t**3
    term2 = P_s * (1 + 3 * (v_u_t / U_r)**2)
    inner = math.sqrt(1 + 0.25 * (v_u_t / V_h)**4) - 0.5 * (v_u_t / V_h)**2
    term3 = P_m * math.sqrt(inner)

    power = term1 + term2 + term3
    return power * delta_t

class UAVISACEnvironment(gym.Env):
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14, delta_t: float = 4.0,
                 E_tot: float = 25000.0, energy_penalty: float = 5.0,
                 # 注意：删除了 normalize_state 参数，因为环境只负责输出原始物理值
                 normalize_reward=True, # 保留奖励缩放
                 eav_agg: str = 'logsumexp', eav_logsumexp_kappa: float = 0.5,
                 eav_threshold: float = 10.0, eav_penalty_coef: float = 5.0, eav_penalty_cap: float = 20.0,
                 comm_penalty_type: str = 'softplus', comm_threshold: float = 10.0, comm_penalty_coef: float = 0.5,
                 comm_softplus_kappa: float = 1.0, comm_huber_delta: float = 1.0,
                 comm_penalty_cap_per_user: float = 5.0, comm_penalty_cap_total: float = 10.0,
                 comm_penalty_avg_over_k: bool = True,
                 action_smooth_coef: float = 0.8, user_move_range: float = 20.0,
                 reward_scale: float = 0.1,
                 eta_clip_max: float = 15.0,
                 comm_penalty_clip_max: float = 5.0,
                 eav_penalty_clip_max: float = 8.0):
        super(UAVISACEnvironment, self).__init__()

        # 参数设置
        self.N = N
        self.K = K
        self.H = H
        self.H1 = H1
        self.sigma2 = sigma2
        self.l_max = l_max
        self.delta_t = delta_t
        self.E_tot = E_tot
        self.energy_penalty = energy_penalty

        # 惩罚项参数
        self.eav_agg = eav_agg
        self.eav_logsumexp_kappa = eav_logsumexp_kappa
        self.eav_threshold = eav_threshold
        self.eav_penalty_coef = eav_penalty_coef
        self.eav_penalty_cap = eav_penalty_cap

        self.comm_penalty_type = comm_penalty_type
        self.comm_threshold = comm_threshold
        self.comm_penalty_coef = comm_penalty_coef
        self.comm_softplus_kappa = comm_softplus_kappa
        self.comm_huber_delta = comm_huber_delta
        self.comm_penalty_cap_per_user = comm_penalty_cap_per_user
        self.comm_penalty_cap_total = comm_penalty_cap_total
        self.comm_penalty_avg_over_k = comm_penalty_avg_over_k
        
        self.action_smooth_coef = action_smooth_coef
        self.user_move_range = user_move_range
        self.reward_scale = reward_scale
        
        self.eta_clip_max = eta_clip_max
        self.comm_penalty_clip_max = comm_penalty_clip_max
        self.eav_penalty_clip_max = eav_penalty_clip_max

        # 空间定义
        self.X_min, self.X_max = -400.0, 400.0
        self.Y_min, self.Y_max = -400.0, 400.0
        self.P_max = 0.1

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 观察空间：(当前状态 + 上一步动作) + (前一帧状态 + 前一帧动作)
        # obs_dim = (uav_pos(2) + users(2*K) + prev_action(3))
        single_obs_dim = 2 + 2 * self.K + 3
        obs_dim = single_obs_dim * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.total_energy = 0.0
        self.current_episode_reward = 0
        self.prev_obs = None
        self.normalize_reward = normalize_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_slot = 0
        self.current_episode_reward = 0
        self.total_energy = 0.0

        # 随机初始化位置
        x1 = self.np_random.uniform(-400, 400)
        y1 = self.np_random.uniform(-400, 400)
        self.uav_position = np.array([x1, y1, self.H])

        self.user_positions = []
        for _ in range(self.K):
            while True:
                user_x = self.np_random.uniform(-400, 400)
                user_y = self.np_random.uniform(-400, 400)
                if (user_x ** 2 + user_y ** 2) > 10000:
                    break
            self.user_positions.append([user_x, user_y, 0])
        self.user_positions = np.array(self.user_positions)

        self.target_position = np.array([100.0, 100.0, self.H1])
        self.radar_receiver_position = np.array([0.0, 0.0, 0.0])
        self.prev_action = np.array([0.0, 0.0, 0.0])

        current_obs = self._get_obs()
        self.prev_obs = np.zeros_like(current_obs) # 初始时刻前一帧为0

        combined_obs = self._get_combined_obs()
        
        # 🔥 修改点：不再进行状态归一化，直接返回物理值
        return combined_obs, {}

    def _get_obs(self):
        obs = np.concatenate([
            self.uav_position[:2], # 只取 x, y
            self.user_positions[:, :2].flatten(),
            self.prev_action
        ])
        return obs

    def _get_combined_obs(self):
        return np.concatenate([self._get_obs(), self.prev_obs])

    def step(self, action):
        # 动作映射
        angle = action[0] * np.pi
        # distance = action[1] * self.l_max
        distance = ((action[1] + 1.0) / 2.0) * self.l_max
        power_allocation = (action[2] + 1) / 2 * self.P_max

        # 物理更新
        delta_x = distance * np.cos(angle)
        delta_y = distance * np.sin(angle)
        new_uav_position = self.uav_position.copy()
        new_uav_position[0] += delta_x
        new_uav_position[1] += delta_y

        terminated = False
        truncated = False
        info = {}

        # 1. 越界检查
        # 1. 计算新位置
        temp_position = self.uav_position.copy()
        temp_position[0] += delta_x
        temp_position[1] += delta_y

        # 2. 截断位置到合法范围内
        clipped_position = np.clip(
            temp_position, 
            [self.X_min, self.Y_min, 0], 
            [self.X_max, self.Y_max, 1000]
        )

        # 3. 检查是否发生了截断（即是否撞墙）
        is_out_of_bound = not np.array_equal(temp_position[:2], clipped_position[:2])

        # 4. 如果撞墙，给予适量惩罚，但允许位置更新到边界处
        if is_out_of_bound:
            # 惩罚可以小一点，因为我们已经物理限制它了，它没占到便宜
            raw_reward = -5.0 # 从 -50 改为 -5
            info['boundary_penalty'] = 5.0
        else:
            # 正常计算奖励
            raw_reward, info_reward = self._calculate_reward(clipped_position, power_allocation)
            # 合并 info
            info.update(info_reward)
            info['boundary_penalty'] = 0.0

        # 5. 更新位置 (重要：即使撞墙也更新到边界，防止陷入死循环)
        self.uav_position = clipped_position

        # 2. 能耗计算
        horizontal_speed = abs(distance) / self.delta_t # 注意原代码除以4可能就是delta_t
        energy_t = calc_energy(horizontal_speed, self.delta_t)
        self.total_energy += energy_t
        
        if self.total_energy > self.E_tot:
            raw_reward -= self.energy_penalty
            info['energy_penalty'] = float(self.energy_penalty)
            # 能量耗尽通常也可以视为 terminated，视需求而定
        else:
            info['energy_penalty'] = 0.0
        
        # 3. 动作平滑
        action_diff = action - self.prev_action
        action_smooth_penalty = self.action_smooth_coef * np.sum(action_diff ** 2)
        raw_reward -= action_smooth_penalty
        info['action_smooth_penalty'] = float(action_smooth_penalty)
        info['reward_final_unscaled'] = float(raw_reward)

        # 4. 奖励缩放
        reward = raw_reward * self.reward_scale
        info['reward_final'] = float(reward)

        self.current_episode_reward += reward
        self.prev_action = action
        self._update_user_positions()
        
        # 5. 时间步更新与截断/终止逻辑
        self.current_slot += 1

        # 🔥 关键修改：区分 Terminated 和 Truncated
        if self.current_slot >= self.N: # N=50
            truncated = True # 时间到了，这叫截断
            terminated = False # 并非物理意义上的失败或完成

        # 获取新状态
        current_obs = self._get_obs()
        combined_obs = self._get_combined_obs()
        self.prev_obs = current_obs.copy()

        # 🔥 不归一化，直接返回
        return combined_obs, reward, terminated, truncated, info

    def _calculate_reward(self, uav_position, power_allocation):
        # ... (保持原有的奖励计算逻辑不变，包括分项裁剪等) ...
        # 为了节省篇幅，这里复用你原有的逻辑
        eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
        eta_0_clipped = np.clip(eta_0, 0.0, self.eta_clip_max)
        reward = eta_0_clipped

        total_comm_penalty = 0.0
        for k in range(self.K):
            distance = np.linalg.norm(uav_position - self.user_positions[k])
            snr = self._calculate_communication_snr(distance, power_allocation)
            snr_gap = self.comm_threshold - snr

            per_user_penalty = 0.0
            if self.comm_penalty_type == 'hinge':
                per_user_penalty = max(0.0, self.comm_penalty_coef * snr_gap)
            elif self.comm_penalty_type == 'huber':
                gap = max(0.0, snr_gap)
                delta = max(float(self.comm_huber_delta), 1e-6)
                if gap <= delta:
                    per_user_penalty = self.comm_penalty_coef * (0.5 * (gap ** 2) / delta)
                else:
                    per_user_penalty = self.comm_penalty_coef * (gap - 0.5 * delta)
            else:
                softplus_gap = np.logaddexp(0.0, self.comm_softplus_kappa * snr_gap) / self.comm_softplus_kappa
                softplus_0 = np.logaddexp(0.0, 0.0) / self.comm_softplus_kappa
                per_user_penalty = self.comm_penalty_coef * (softplus_gap - softplus_0)
                per_user_penalty = max(0.0, per_user_penalty)

            total_comm_penalty += min(per_user_penalty, self.comm_penalty_cap_per_user)

        if self.comm_penalty_avg_over_k and self.K > 0:
            total_comm_penalty /= float(self.K)
        comm_penalty = min(total_comm_penalty, self.comm_penalty_cap_total)
        comm_penalty_clipped = np.clip(comm_penalty, 0.0, self.comm_penalty_clip_max)
        reward -= comm_penalty_clipped

        eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(uav_position, power_allocation)
        if len(eavesdropper_snr_list) == 0:
            sensing_snr_eavesdropper = 0.0
        elif self.eav_agg == 'max':
            sensing_snr_eavesdropper = float(np.max(np.array(eavesdropper_snr_list)))
        elif self.eav_agg == 'logsumexp':
            x = np.array(eavesdropper_snr_list)
            kappa = float(self.eav_logsumexp_kappa)
            m = float(np.max(x))
            sensing_snr_eavesdropper = m + (1.0 / kappa) * float(np.log(np.sum(np.exp(kappa * (x - m)))))
        else:
            if len(eavesdropper_snr_list) >= 2:
                top2 = np.partition(np.array(eavesdropper_snr_list), -2)[-2:]
                sensing_snr_eavesdropper = float(np.mean(top2))
            else:
                sensing_snr_eavesdropper = float(eavesdropper_snr_list[0])

        snr_gap2 = sensing_snr_eavesdropper - self.eav_threshold
        eav_penalty = 0.0
        if snr_gap2 > 0:
            eav_penalty = min(self.eav_penalty_coef * snr_gap2, self.eav_penalty_cap)
            eav_penalty_clipped = np.clip(eav_penalty, 0.0, self.eav_penalty_clip_max)
            reward -= eav_penalty_clipped

        info = {
            'eta_0': float(eta_0),
            'eta_0_clipped': float(eta_0_clipped),
            'comm_penalty': float(comm_penalty),
            'comm_penalty_clipped': float(comm_penalty_clipped),
            'snr_gap2':float(snr_gap2),
            'eav_penalty': float(eav_penalty),
            'eav_penalty_clipped': float(eav_penalty_clipped) if snr_gap2 > 0 else 0.0,
            'energy_penalty': 0.0,
            'boundary_penalty': 0.0,
            'reward_raw': float(reward),
        }
        return reward, info

    # 保持原有的物理计算函数 _calculate_communication_snr, _calculate_sensing_snr_legal 等不变
    def _calculate_communication_snr(self, distance, power_allocation):
        c1 = 12.081
        c2 = 0.11395
        mu_los = 1.44544
        mu_nlos = 199.526
        fc = 2.4e9
        c = 3e8
        alpha = 2.0
        K_0 = (4 * np.pi * fc) / c
        d_3d = np.sqrt(self.H**2 + distance**2)
        theta = np.arcsin(self.H / d_3d) * 180 / np.pi
        p_los = 1 / (1 + c1 * np.exp(-c2 * (theta - c1)))
        p_nlos = 1 - p_los
        los = mu_los * (K_0 * d_3d) ** alpha
        nlos = mu_nlos * (K_0 * d_3d) ** alpha
        los = 10 * np.log10(max(los, 1e-5))
        nlos = 10 * np.log10(max(nlos, 1e-5))
        L = 10 ** ((p_los * los + p_nlos * nlos) / 10)
        omega = 1 / L
        snr = (omega * power_allocation) / self.sigma2
        return 10 * np.log10(max(snr, 1e-5))

    def _calculate_sensing_snr_legal(self, uav_position, power_allocation):
        d_t = np.linalg.norm(uav_position - self.target_position)
        d_r = np.linalg.norm(self.target_position - self.radar_receiver_position)
        G_tx = 13
        G_rx = 13
        c = 3e8
        fc = 2.4e9
        lambda_c = c / fc
        sigma = 1.0
        P_r = (power_allocation * 10 ** (G_tx / 10) * 10 ** (G_rx / 10) * lambda_c**2 * sigma) / \
              (((4 * np.pi)**3) * d_t**2 * d_r**2)
        return 10 * np.log10(max(P_r / self.sigma2, 1e-5))

    def _calculate_sensing_snr_eavesdropper(self, uav_position, power_allocation):
        eavesdropper_snr_list = []
        d_t = np.linalg.norm(uav_position - self.target_position)
        G_tx, G_rx = 13, 13
        lambda_c = 3e8 / 2.4e9
        for k in range(self.K):
            d_k_r = np.linalg.norm(self.target_position - self.user_positions[k])
            P_r_k = (power_allocation *10 ** (G_tx / 10) * 10 ** (G_rx / 10) * lambda_c**2 * 1.0) / \
                    (((4 * np.pi)**3) * d_t**2 * d_k_r**2)
            eavesdropper_snr_list.append(10 * np.log10(max(P_r_k / self.sigma2, 1e-5)))
        return eavesdropper_snr_list

    def _update_user_positions(self):
        for k in range(self.K):
            original_x = self.user_positions[k, 0]
            original_y = self.user_positions[k, 1]
            valid = False
            while not valid:
                move_distance = self.np_random.uniform(0, self.user_move_range)
                move_angle = self.np_random.uniform(-np.pi, np.pi)
                new_x = original_x + move_distance * np.cos(move_angle)
                new_y = original_y + move_distance * np.sin(move_angle)
                distance_sq = new_x ** 2 + new_y ** 2
                if (-400 <= new_x <= 400) and (-400 <= new_y <= 400) and (distance_sq > 10000):
                    valid = True
            self.user_positions[k, 0] = new_x
            self.user_positions[k, 1] = new_y