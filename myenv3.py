import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from gymnasium.envs.registration import register
register(
    id='Env',
    entry_point='myenv3:UAVISACEnvironment', # 注意：这里假设文件名是 myenv3.py
    max_episode_steps=50
)

def calc_energy(v_u_t: float, delta_t: float) -> float:
    """根据论文公式计算能耗 (J)。"""
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
# 主环境类
# ============================================================
class UAVISACEnvironment(gym.Env):
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14, delta_t: float = 4.0,
                 E_tot: float = 25000.0, energy_penalty: float = 5.0,
                 use_state_scaling=True,
                 
                 # 感知与安全参数
                 eav_threshold: float = 10.0, 
                 eav_penalty_coef: float = 2.0, 
                 eav_penalty_clip_max: float = 1000.0, # 默认设极大，实际逻辑中移除截断
                 
                 # 通信参数 (优先级较低)
                 comm_threshold=10.0,
                 comm_penalty_coef=1.5,
                 comm_softplus_kappa: float = 5.0, # 稍微增加陡峭度
                 comm_penalty_clip_per_user=15.0,
                 comm_penalty_clip_total=30.0,

                 action_smooth_coef: float = 0.1, 
                 user_move_range: float = 20.0,
                 reward_scale: float = 0.1,       
                 ): 
                 
        super(UAVISACEnvironment, self).__init__()

        # 基础参数
        self.N, self.K = N, K
        self.H, self.H1 = H, H1
        self.sigma2 = sigma2
        self.l_max = l_max
        self.delta_t = delta_t
        self.E_tot = E_tot
        self.energy_penalty = energy_penalty

        # 核心奖励参数
        self.eav_threshold = eav_threshold
        self.eav_penalty_coef = eav_penalty_coef
        self.eav_penalty_clip_max = eav_penalty_clip_max # 实际上不再作为硬截断使用

        self.comm_threshold = comm_threshold
        self.comm_penalty_coef = comm_penalty_coef
        self.comm_softplus_kappa = comm_softplus_kappa
        self.comm_penalty_clip_per_user = comm_penalty_clip_per_user
        self.comm_penalty_clip_total = comm_penalty_clip_total
        
        self.action_smooth_coef = action_smooth_coef
        self.user_move_range = user_move_range
        self.reward_scale = reward_scale

        # 空间限制
        self.X_min, self.X_max = -400.0, 400.0
        self.Y_min, self.Y_max = -400.0, 400.0

        # 动作与观测
        self.P_max = 0.1
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        obs_dim = (2 + 2 * self.K + 3) * 2
        self.use_state_scaling = use_state_scaling
        if self.use_state_scaling:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.total_energy = 0.0
        self.current_episode_reward = 0
        self.prev_obs = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_slot = 0
        self.current_episode_reward = 0
        self.total_energy = 0.0

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

        self.prev_obs = np.zeros(2 + 2 * self.K + 3, dtype=np.float32)
        combined_obs = self._get_combined_obs()
        
        return combined_obs, {}

    def _get_raw_obs(self):
        """Return raw single-frame observation (unscaled)."""
        return np.concatenate([
            self.uav_position[:2],
            self.user_positions[:, :2].flatten(),
            self.prev_action
        ])

    def _scale_obs(self, obs):
        """Apply fixed manual scaling to a single-frame observation.
        
        Coordinates (UAV x/y, users x/y): divide by 400.0 → [-1, 1]
        prev_action: already in [-1, 1], kept as-is.
        """
        scaled = obs.copy().astype(np.float32)
        coord_end = 2 + 2 * self.K
        scaled[:coord_end] /= 400.0
        return scaled

    def _get_combined_obs(self):
        """Return combined observation [current, prev], with optional fixed scaling."""
        raw = self._get_raw_obs()
        if self.use_state_scaling:
            current = self._scale_obs(raw)
            prev = self._scale_obs(self.prev_obs)
        else:
            current = raw.astype(np.float32)
            prev = self.prev_obs.astype(np.float32)
        return np.concatenate([current, prev])

    def step(self, action):
        angle = action[0] * np.pi
        distance = action[1] * self.l_max
        power_allocation = (action[2] + 1) / 2 * self.P_max

        delta_x = distance * np.cos(angle)
        delta_y = distance * np.sin(angle)
        new_uav_position = self.uav_position.copy()
        new_uav_position[0] += delta_x
        new_uav_position[1] += delta_y

        # 边界处理
        if new_uav_position[0] < self.X_min or new_uav_position[0] > self.X_max or \
           new_uav_position[1] < self.Y_min or new_uav_position[1] > self.Y_max:
            raw_reward = -100.0 # 越界重罚
            info = {
                'eta_0': 0.0, 'comm_penalty': 0.0, 'eav_penalty': 0.0,
                'reward_final': raw_reward * self.reward_scale,
                'leakage_count': 0,
                'total_users': self.K,
            }
        else:
            # 正常计算奖励
            raw_reward, info = self._calculate_reward(new_uav_position, power_allocation)
            self.uav_position = new_uav_position

        # 能耗惩罚
        horizontal_speed = abs(distance) / 4.0
        energy_t = calc_energy(horizontal_speed, self.delta_t)
        self.total_energy += energy_t
        if self.total_energy > self.E_tot:
            raw_reward -= self.energy_penalty
        
        # 动作平滑惩罚
        action_diff = action - self.prev_action
        action_smooth_penalty = self.action_smooth_coef * np.sum(action_diff ** 2)
        raw_reward -= action_smooth_penalty

        info['reward_final_unscaled'] = float(raw_reward)
        
        # 最终缩放
        reward = raw_reward * self.reward_scale
        info['reward_final'] = float(reward)

        self.current_episode_reward += reward
        self.prev_action = action
        self._update_user_positions()
        self.current_slot += 1

        done = (self.current_slot == 50)
        
        current_obs = self._get_raw_obs()
        combined_obs = self._get_combined_obs()
        self.prev_obs = current_obs.copy()

        return combined_obs, reward, done, False, info

    def _calculate_reward(self, uav_position, power_allocation):
        # 1. 感知收益 (Sensing Gain)
        eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
        # 对感知奖励进行软截断，防止正向奖励爆炸引导 Agent 走极端，但允许优秀表现
        # 假设 30dB 已经是极好的值
        R_sense = min(eta_0, 30.0)
        
        # 2. 安全/窃听惩罚 (Security Penalty) - 核心优先项
        eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(uav_position, power_allocation)
        R_eav = 0.0
        eav_penalty_raw = 0.0
        
        # 感知泄漏率统计：统计有多少用户的窃听SNR超过阈值
        leakage_count = 0
        total_users = self.K
        
        if len(eavesdropper_snr_list) > 0:
            eav_snrs = np.array(eavesdropper_snr_list, dtype=np.float32)
            
            # 统计泄漏用户数（SNR超过阈值的用户）
            leakage_count = int(np.sum(eav_snrs > self.eav_threshold))
            
            # 使用 Max 策略，只要有一个窃听者超标，就算违规
            max_eav_snr = np.max(eav_snrs)
            
            # 计算 Gap
            snr_gap_eav = max_eav_snr - self.eav_threshold
            
            # 关键修改：移除截断，使用 Softplus
            # 使用 comm_softplus_kappa 或默认 2.0，让惩罚更陡峭
            kappa = 2.0 
            eav_penalty_raw = np.logaddexp(0.0, kappa * snr_gap_eav) / kappa
            
            # 乘上系数 (该系数在 auto.py 中会被校准得很大)
            R_eav = eav_penalty_raw * self.eav_penalty_coef

        # 3. 通信惩罚 (Communication Penalty) - 次要项
        R_comm = 0.0
        avg_comm_penalty = 0.0
        if self.K > 0:
            comm_penalties = []
            for k in range(self.K):
                distance = np.linalg.norm(uav_position - self.user_positions[k])
                snr = self._calculate_communication_snr(distance, power_allocation)
                snr_gap = self.comm_threshold - snr
                
                # Softplus
                p_smooth = np.logaddexp(0.0, self.comm_softplus_kappa * snr_gap) / self.comm_softplus_kappa
                comm_penalties.append(p_smooth)
            
            avg_comm_penalty = np.mean(comm_penalties)
            # 对通信惩罚可以保留一个较宽的截断，避免它掩盖了安全惩罚
            avg_comm_penalty_clipped = min(avg_comm_penalty, self.comm_penalty_clip_total)
            
            R_comm = avg_comm_penalty_clipped * self.comm_penalty_coef

        # 总奖励
        reward = R_sense - R_eav - R_comm

        info = {
            'eta_0': float(eta_0),
            'eta_0_clipped': float(min(eta_0, 30.0)),  # 添加裁剪后的感知SNR
            'eav_penalty_raw': float(eav_penalty_raw),
            'eav_penalty_weighted': float(R_eav),
            'eav_penalty': float(eav_penalty_raw),  # 对齐main.py的命名
            'eav_penalty_clipped': float(eav_penalty_raw),  # myenv3中没有硬裁剪
            'comm_penalty': float(avg_comm_penalty),
            'comm_penalty_clipped': float(avg_comm_penalty_clipped),  # 添加裁剪后的通信惩罚
            'reward_raw': float(reward),
            'leakage_count': leakage_count,
            'total_users': total_users,
            'eavesdropper_snr_list': eavesdropper_snr_list,  # 窃听者SNR列表 (dB)
        }
        return reward, info

    # --- 物理公式计算函数 (保持原样) ---
    def _calculate_communication_snr(self, distance, power_allocation):
        c1, c2 = 12.081, 0.11395
        mu_los, mu_nlos = 1.44544, 199.526
        fc, c, alpha = 2.4e9, 3e8, 2.0
        K_0 = (4 * np.pi * fc) / c
        d_3d = np.sqrt(self.H**2 + distance**2)
        theta = np.arcsin(self.H / d_3d) * 180 / np.pi
        p_los = 1 / (1 + c1 * np.exp(-c2 * (theta - c1)))
        p_nlos = 1 - p_los
        los = mu_los * (K_0 * d_3d) ** alpha
        nlos = mu_nlos * (K_0 * d_3d) ** alpha
        L = 10 ** ((p_los * 10*np.log10(max(los,1e-5)) + p_nlos * 10*np.log10(max(nlos,1e-5))) / 10)
        snr = ((1.0/L) * power_allocation) / self.sigma2
        return 10 * np.log10(max(snr, 1e-10))

    def _calculate_sensing_snr_legal(self, uav_position, power_allocation):
        d_t = np.linalg.norm(uav_position - self.target_position)
        d_r = np.linalg.norm(self.target_position - self.radar_receiver_position)
        G_tx, G_rx, lambda_c, sigma = 13, 13, 3e8/2.4e9, 1.0
        P_r = (power_allocation * 10**(G_tx/10) * 10**(G_rx/10) * lambda_c**2 * sigma) / \
              (((4 * np.pi)**3) * max(d_t**2, 1e-5) * max(d_r**2, 1e-5))
        return 10 * np.log10(max(P_r / self.sigma2, 1e-10))

    def _calculate_sensing_snr_eavesdropper(self, uav_position, power_allocation):
        snr_list = []
        d_t = np.linalg.norm(uav_position - self.target_position)
        G_tx, G_rx, lambda_c, sigma = 13, 13, 3e8/2.4e9, 1.0
        for k in range(self.K):
            d_k_r = np.linalg.norm(self.target_position - self.user_positions[k])
            P_r_k = (power_allocation * 10**(G_tx/10) * 10**(G_rx/10) * lambda_c**2 * sigma) / \
                    (((4 * np.pi)**3) * max(d_t**2, 1e-5) * max(d_k_r**2, 1e-5))
            snr_list.append(10 * np.log10(max(P_r_k / self.sigma2, 1e-10)))
        return snr_list

    def _update_user_positions(self):
        for k in range(self.K):
            while True:
                move_d = self.np_random.uniform(0, self.user_move_range)
                move_a = self.np_random.uniform(-np.pi, np.pi)
                nx = self.user_positions[k, 0] + move_d * np.cos(move_a)
                ny = self.user_positions[k, 1] + move_d * np.sin(move_a)
                if (-400 <= nx <= 400) and (-400 <= ny <= 400) and (nx**2 + ny**2 > 10000):
                    self.user_positions[k, 0] = nx
                    self.user_positions[k, 1] = ny
                    break