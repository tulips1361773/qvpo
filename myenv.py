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
# 改进版：状态归一化器类（完整版）
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
        
        # ✅ 新增：训练模式开关
        self.training = True
        
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
        
    def normalize(self, state, update_stats=None):
        """
        归一化状态
        
        Args:
            state: 输入状态
            update_stats: 
                - None: 根据 self.training 决定是否更新统计量
                - True: 强制更新统计量
                - False: 强制不更新统计量
        """
        # ✅ 改进点1：根据训练模式自动决定是否更新
        if update_stats is None:
            update_stats = self.training
        
        state = np.asarray(state, dtype=np.float32)
        
        if update_stats:
            self.update(state)
        
        # ✅ 改进点2：动态epsilon，初期更保守
        adaptive_epsilon = max(self.epsilon, 0.01 / (1 + self.count/1000))
        std = np.sqrt(self.var) + adaptive_epsilon
        normalized = (state - self.mean) / std
        
        # Clip 防止极端值
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return normalized.astype(np.float32)
    
    # ✅ 新增：模式切换方法
    def set_training(self, mode):
        """
        切换训练/评估模式
        
        Args:
            mode: True表示训练模式，False表示评估模式
        """
        self.training = mode


# ============================================================
# 主环境类（已集成归一化功能）
# ============================================================
class UAVISACEnvironment(gym.Env):
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14, delta_t: float = 4.0,
                 E_tot: float = 25000.0, energy_penalty: float = 5.0,  # 降低能量阈值使其生效
                 normalize_state=True, normalize_reward=True,
                 eav_agg: str = 'logsumexp', eav_logsumexp_kappa: float = 0.5,  # 建议2: 使用logsumexp平滑聚合，降低kappa
                 eav_threshold: float = 10.0, eav_penalty_coef: float = 1.0, eav_penalty_cap: float = 10.0,  # 降低惩罚系数
                 comm_penalty_type: str = 'softplus', comm_threshold: float = 10.0, comm_penalty_coef: float = 0.5,  # 降低惩罚系数
                 comm_softplus_kappa: float = 1.0, comm_huber_delta: float = 1.0,  # 降低kappa
                 comm_penalty_cap_per_user: float = 5.0, comm_penalty_cap_total: float = 10.0,  # 降低cap
                 comm_penalty_avg_over_k: bool = True,
                 action_smooth_coef: float = 0.8, user_move_range: float = 20.0,  # 增大动作平滑惩罚权重 0.3→0.8
                 reward_scale: float = 0.1,  # 奖励缩放因子
                 # 建议3: 分项裁剪参数
                 eta_clip_max: float = 15.0,  # 感知SNR裁剪上限
                 comm_penalty_clip_max: float = 5.0,  # 通信惩罚裁剪上限
                 eav_penalty_clip_max: float = 5.0):  # 窃听惩罚裁剪上限
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
        
        # 新增参数
        self.action_smooth_coef = action_smooth_coef
        self.user_move_range = user_move_range
        self.reward_scale = reward_scale
        
        # 建议3: 分项裁剪参数
        self.eta_clip_max = eta_clip_max
        self.comm_penalty_clip_max = comm_penalty_clip_max
        self.eav_penalty_clip_max = eav_penalty_clip_max

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

        # 初始化环境
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_slot = 0
        self.current_episode_reward = 0
        self.total_energy = 0.0

        x1 = self.np_random.uniform(-400, 400)
        y1 = self.np_random.uniform(-400, 400)
        self.uav_position = np.array([x1, y1, self.H])

        # 初始化通信用户位置
        self.user_positions = []
        for _ in range(self.K):
            while True:
                user_x = self.np_random.uniform(-400, 400)
                user_y = self.np_random.uniform(-400, 400)
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

        # 返回组合后的观察值（已归一化）
        combined_obs = self._get_combined_obs()
        
        # 🔥 状态归一化
        if self.normalize_state:
            combined_obs = self.state_normalizer.normalize(combined_obs)
        
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
            info = {
                'eta_0': 0.0,
                'comm_penalty': 0.0,
                'eav_penalty': 0.0,
                'energy_penalty': 0.0,
                'boundary_penalty': 20.0,
                'reward_raw': -20.0,
            }
        else:
            reward, info = self._calculate_reward(new_uav_position, power_allocation)
            self.uav_position = new_uav_position

         # solu4: 修改第一次裁剪范围
        reward = np.clip(reward, -20.0, 30.0)
        info['reward_clip_1'] = float(reward)

        # 能耗计算
        horizontal_speed = abs(distance) / 4.0
        energy_t = calc_energy(horizontal_speed, self.delta_t)
        self.total_energy += energy_t
        if self.total_energy > self.E_tot:
            reward -= self.energy_penalty
            info['energy_penalty'] = float(self.energy_penalty)
        else:
            info['energy_penalty'] = 0.0
        
        # 新增：动作平滑惩罚（抑制Bang-Bang控制）
        action_diff = action - self.prev_action
        action_smooth_penalty = self.action_smooth_coef * np.sum(action_diff ** 2)
        reward -= action_smooth_penalty
        info['action_smooth_penalty'] = float(action_smooth_penalty)

        # solu4: 删除第二次裁剪
        # reward = np.clip(reward, -30.0, 50.0)
        
        # 奖励缩放（使奖励范围更适合RL训练）
        reward = reward * self.reward_scale
        info['reward_final'] = float(reward)

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
        
        # 获取当前观察值
        current_obs = self._get_obs()

        # 组合当前观察和前一个观察
        combined_obs = self._get_combined_obs()

        # 🔥 状态归一化
        if self.normalize_state:
            combined_obs = self.state_normalizer.normalize(combined_obs)

        # 更新前一个观察值
        self.prev_obs = current_obs.copy()

        return combined_obs, reward, done, False, info

    def _calculate_reward(self, uav_position, power_allocation):
        eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
        
        # 建议3: 分项裁剪 - 对感知SNR进行裁剪，避免极高SNR的边际收益过大
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
        # 建议3: 分项裁剪 - 对通信惩罚进行裁剪
        comm_penalty_clipped = np.clip(comm_penalty, 0.0, self.comm_penalty_clip_max)
        reward -= comm_penalty_clipped

        eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(uav_position, power_allocation)
        if len(eavesdropper_snr_list) == 0:
            sensing_snr_eavesdropper = 0.0
        elif self.eav_agg == 'max':
            sensing_snr_eavesdropper = float(np.max(np.array(eavesdropper_snr_list, dtype=np.float32)))
        elif self.eav_agg == 'logsumexp':
            x = np.array(eavesdropper_snr_list, dtype=np.float32)
            kappa = float(self.eav_logsumexp_kappa)
            kappa = max(kappa, 1e-6)
            m = float(np.max(x))
            sensing_snr_eavesdropper = m + (1.0 / kappa) * float(np.log(np.sum(np.exp(kappa * (x - m)))))
        else:
            if len(eavesdropper_snr_list) >= 2:
                top2 = np.partition(np.array(eavesdropper_snr_list, dtype=np.float32), -2)[-2:]
                sensing_snr_eavesdropper = float(np.mean(top2))
            else:
                sensing_snr_eavesdropper = float(eavesdropper_snr_list[0])

        snr_gap2 = sensing_snr_eavesdropper - self.eav_threshold
        eav_penalty = 0.0
        if snr_gap2 > 0:
            eav_penalty = min(self.eav_penalty_coef * snr_gap2, self.eav_penalty_cap)
            # 建议3: 分项裁剪 - 对窃听惩罚进行裁剪
            eav_penalty_clipped = np.clip(eav_penalty, 0.0, self.eav_penalty_clip_max)
            reward -= eav_penalty_clipped

        info = {
            'eta_0': float(eta_0),
            'eta_0_clipped': float(eta_0_clipped),  # 建议3: 裁剪后的感知SNR
            'comm_penalty': float(comm_penalty),
            'comm_penalty_clipped': float(comm_penalty_clipped),  # 建议3: 裁剪后的通信惩罚
            'eav_penalty': float(eav_penalty),
            'eav_penalty_clipped': float(eav_penalty_clipped) if snr_gap2 > 0 else 0.0,  # 建议3: 裁剪后的窃听惩罚
            'energy_penalty': 0.0,
            'boundary_penalty': 0.0,
            'reward_raw': float(reward),
        }

        return reward, info

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
                move_distance = self.np_random.uniform(0, self.user_move_range)  # 使用可配置的移动范围
                move_angle = self.np_random.uniform(-np.pi, np.pi)

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