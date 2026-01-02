import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from gymnasium.envs.registration import register
register(
    id='Env',
    entry_point='doubleobservation:UAVISACEnvironment',
    max_episode_steps=50
)

def calc_energy(
    v_u_t: float,
    delta_t: float
) -> float:
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


# ============================================================
# 新增：状态归一化器类
# ============================================================
class StateNormalizer:
    """在线运行时状态归一化器（Running Mean & Std）"""
    def __init__(self, state_dim, epsilon=1e-8, clip_range=10.0):
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # 统计量
        self.mean = np.zeros(state_dim, dtype=np.float64)
        self.var = np.ones(state_dim, dtype=np.float64)
        self.count = epsilon  # 避免除零
        
    def update(self, state):
        """更新统计量（Welford's online algorithm）"""
        state = np.asarray(state, dtype=np.float64)
        batch_mean = state
        batch_var = np.zeros_like(state)
        batch_count = 1
        
        # 增量更新
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
        
    def normalize(self, state, update_stats=True):
        """归一化状态"""
        state = np.asarray(state, dtype=np.float32)
        
        if update_stats:
            self.update(state)
        
        # 归一化
        normalized = (state - self.mean) / (np.sqrt(self.var) + self.epsilon)
        
        # Clip 防止极端值
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return normalized.astype(np.float32)


# ============================================================
# 新增：奖励缩放器类
# ============================================================
class RewardScaler:
    """奖励标准化/缩放器"""
    def __init__(self, gamma=0.99, epsilon=1e-8, clip_range=10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # 统计量
        self.return_val = 0.0
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        
    def update(self, reward):
        """更新回报统计"""
        self.return_val = reward + self.gamma * self.return_val
        
        # 增量更新均值和方差
        self.count += 1
        delta = self.return_val - self.mean
        self.mean += delta / self.count
        delta2 = self.return_val - self.mean
        self.var += delta * delta2
        
    def scale(self, reward, update_stats=True):
        """缩放奖励"""
        if update_stats:
            self.update(reward)
        
        # 标准化
        std = np.sqrt(self.var / self.count + self.epsilon)
        scaled_reward = reward / (std + self.epsilon)
        
        # Clip 防止极端值
        scaled_reward = np.clip(scaled_reward, -self.clip_range, self.clip_range)
        
        return scaled_reward
    
    def reset_return(self):
        """每个 episode 结束后重置回报"""
        self.return_val = 0.0


# ============================================================
# 主环境类（已集成归一化功能）
# ============================================================
class UAVISACEnvironment(gym.Env):
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14, delta_t: float = 4.0,
                 E_tot: float = 600000.0, energy_penalty: float = 10.0,
                 normalize_state=True, normalize_reward=True):  # 🔥 新增开关
        super(UAVISACEnvironment, self).__init__()

        # 时间设置
        self.N = N
        self.K = K
        self.H = H
        self.H1 = H1
        self.sigma2 = sigma2
        self.l_max = l_max

        self.delta_t = delta_t
        self.E_tot = E_tot
        self.energy_penalty = energy_penalty

        # 无人机飞行范围约束
        self.X_min = -400.0
        self.X_max = 400.0
        self.Y_min = -400.0
        self.Y_max = 400.0

        # 最大发射功率
        self.P_max = 0.1

        # 动作空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 观察空间
        obs_dim = (2 + 2 * self.K + 3) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.total_energy = 0.0
        self.t1 = 0
        self.rresult = 0
        self.current_episode_reward = 0
        self.episode_rewards = []

        # 初始化观察缓存
        self.prev_obs = None

        # 🔥 归一化开关与归一化器实例
        self.normalize_state = normalize_state
        self.normalize_reward = normalize_reward
        
        if self.normalize_state:
            self.state_normalizer = StateNormalizer(state_dim=obs_dim)
        
        if self.normalize_reward:
            self.reward_scaler = RewardScaler(gamma=0.99)

        # 初始化环境
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.current_slot = 0
        self.current_episode_reward = 0
        self.total_energy = 0.0

        x1 = np.random.uniform(-400, 400)
        y1 = np.random.uniform(-400, 400)
        self.uav_position = np.array([x1, y1, self.H])

        # 初始化通信用户位置
        self.user_positions = []
        for _ in range(self.K):
            while True:
                user_x = np.random.uniform(-400, 400)
                user_y = np.random.uniform(-400, 400)
                if (user_x ** 2 + user_y ** 2) > 10000:
                    break
            self.user_positions.append([user_x, user_y, 0])

        self.user_positions = np.array(self.user_positions)

        # 初始化目标位置
        self.target_position = np.array([100.0, 100.0, self.H1])

        # 初始化合法雷达接收器位置
        self.radar_receiver_position = np.array([0.0, 0.0, 0.0])

        # 初始化上一步动作为零
        self.prev_action = np.array([0.0, 0.0, 0.0])

        # 获取当前观察值
        current_obs = self._get_obs()

        # 初始化前一个观察值为全零
        self.prev_obs = np.zeros_like(current_obs)

        # 🔥 重置奖励归一化器的回报
        if self.normalize_reward:
            self.reward_scaler.reset_return()

        # 返回组合后的观察值（已归一化）
        combined_obs = self._get_combined_obs()
        
        # 🔥 状态归一化
        if self.normalize_state:
            combined_obs = self.state_normalizer.normalize(combined_obs, update_stats=True)
        
        return combined_obs, {}

    def _get_obs(self):
        obs = np.concatenate([
            self.uav_position[:2],
            self.user_positions[:, :2].flatten(),
            self.prev_action
        ])
        return obs

    def _get_combined_obs(self):
        return np.concatenate([self._get_obs(), self.prev_obs])

    def step(self, action):
        # 将动作从 [-1, 1] 映射到实际值
        angle = action[0] * np.pi
        distance = action[1] * self.l_max
        power_allocation = (action[2] + 1) / 2 * self.P_max

        # 更新无人机位置
        delta_x = distance * np.cos(angle)
        delta_y = distance * np.sin(angle)
        new_uav_position = self.uav_position.copy()
        new_uav_position[0] += delta_x
        new_uav_position[1] += delta_y

        # 确保无人机位置在允许范围内
        if new_uav_position[0] < self.X_min or new_uav_position[0] > self.X_max or new_uav_position[1] < self.Y_min or \
                new_uav_position[1] > self.Y_max:
            reward = -20.0
        else:
            reward = self._calculate_reward(new_uav_position, power_allocation)
            self.uav_position = new_uav_position

        # 能耗计算
        horizontal_speed = abs(distance) / 4.0
        energy_t = calc_energy(horizontal_speed, self.delta_t)
        self.total_energy += energy_t
        if self.total_energy > self.E_tot:
            reward -= self.energy_penalty

        # 🔥 奖励缩放
        if self.normalize_reward:
            reward = self.reward_scaler.scale(reward, update_stats=True)

        # 计算奖励
        self.current_episode_reward += reward

        # 记录当前动作
        self.prev_action = action

        # 更新通信用户位置
        self._update_user_positions()

        # 增加时间步
        self.current_slot += 1

        done = False
        if self.current_slot == 50:
            done = True
            if self.t1 < 500:
                self.rresult += self.current_episode_reward / 50
                self.t1 += 1
            else:
                average_rresult = self.rresult / 500
                self.episode_rewards.append(average_rresult)
                self.rresult = 0
                self.t1 = 0
            
            # 🔥 Episode 结束后重置回报
            if self.normalize_reward:
                self.reward_scaler.reset_return()

        # 获取当前观察值
        current_obs = self._get_obs()

        # 组合当前观察和前一个观察
        combined_obs = self._get_combined_obs()

        # 🔥 状态归一化
        if self.normalize_state:
            combined_obs = self.state_normalizer.normalize(combined_obs, update_stats=True)

        # 更新前一个观察值
        self.prev_obs = current_obs.copy()

        return combined_obs, reward, done, False, {}

    def _calculate_reward(self, uav_position, power_allocation):
        # 计算合法接收机的感知信噪比
        eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
        reward = eta_0
        
        communication_threshold = 10.0
        eavesdropper_threshold = 10.0

        # 计算通信用户的通信信噪比
        for k in range(self.K):
            distance = np.linalg.norm(uav_position - self.user_positions[k])
            snr = self._calculate_communication_snr(distance, power_allocation)
            snr_gap = communication_threshold - snr
            if snr_gap > 0:
                reward -= 2*snr_gap

        # 计算非法感知者的感知信噪比
        eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(uav_position, power_allocation)
        sensing_snr_eavesdropper = max(eavesdropper_snr_list)
        snr_gap2 = sensing_snr_eavesdropper - eavesdropper_threshold
        if snr_gap2 > 0:
            reward -= 5*snr_gap2

        return reward

    def _calculate_communication_snr(self, distance, power_allocation):
        # 通信模型中的信道增益计算
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
        los = max(los, 1e-5)
        nlos = max(nlos, 1e-5)
        los = 10 * np.log10(los)
        nlos = 10 * np.log10(nlos)

        L = p_los * los + p_nlos * nlos
        L = 10 ** (L / 10)
        omega = 1 / L

        snr = (omega * power_allocation) / self.sigma2
        snr = max(snr, 1e-5)
        snr_db = 10 * np.log10(snr)

        return snr_db

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

        snr = P_r / self.sigma2
        snr = max(snr, 1e-5)
        snr_db = 10 * np.log10(snr)
        return snr_db

    def _calculate_sensing_snr_eavesdropper(self, uav_position, power_allocation):
        eavesdropper_snr_list = []
        d_t = np.linalg.norm(uav_position - self.target_position)

        G_tx = 13
        G_rx = 13
        c = 3e8
        fc = 2.4e9
        lambda_c = c / fc
        sigma = 1.0

        for k in range(self.K):
            d_k_r = np.linalg.norm(self.target_position - self.user_positions[k])

            P_r_k = (power_allocation *10 ** (G_tx / 10) * 10 ** (G_rx / 10) * lambda_c**2 * sigma) / \
                    (((4 * np.pi)**3) * d_t**2 * d_k_r**2)

            snr_k = P_r_k / self.sigma2
            snr_k = max(snr_k, 1e-5)
            snr_db_k = 10 * np.log10(snr_k)
            eavesdropper_snr_list.append(snr_db_k)

        return eavesdropper_snr_list

    def _update_user_positions(self):
        for k in range(self.K):
            original_x = self.user_positions[k, 0]
            original_y = self.user_positions[k, 1]
            valid = False

            while not valid:
                move_distance = np.random.uniform(0, 50)
                move_angle = np.random.uniform(-np.pi, np.pi)

                delta_x = move_distance * np.cos(move_angle)
                delta_y = move_distance * np.sin(move_angle)

                new_x = original_x + delta_x
                new_y = original_y + delta_y

                distance_sq = new_x ** 2 + new_y ** 2

                coord_in_range = (-400 <= new_x <= 400) and (-400 <= new_y <= 400)
                safe_distance = distance_sq > 10000

                valid = coord_in_range and safe_distance

            self.user_positions[k, 0] = new_x
            self.user_positions[k, 1] = new_y