import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from gymnasium.envs.registration import register            # 注册一个自定义的强化学习环境
register(
    id='Env',  # 这里用你实际的环境ID
    entry_point='doubleobservation:UAVISACEnvironment',     # 这个路径应该是实际模块的路径，是环境类所在的模块
    max_episode_steps=50  # 根据需要调整
)
def calc_energy(
    v_u_t: float,   # 无人机在当前时隙的飞行速度 (m/s)
    delta_t: float  # 时隙长度 (s)
) -> float:         # 返回类型提示
    """根据论文公式(13) 计算单个时隙能耗 (J)。参数固定在类中给出。"""
    d_0   = 0.6
    rho_a = 1.225
    z     = 0.05
    G     = 0.503
    P_s   = 79.85
    U_r   = 120.0
    P_m   = 88.63
    V_h   = 4.03

    # 第一项：气动阻力功率
    term1 = 0.5 * d_0 * rho_a * z * G * v_u_t**3

    # 第二项：姿态功率
    term2 = P_s * (1 + 3 * (v_u_t / U_r)**2)

    # 第三项：机动功率
    inner = math.sqrt(1 + 0.25 * (v_u_t / V_h)**4) - 0.5 * (v_u_t / V_h)**2
    term3 = P_m * math.sqrt(inner)

    power = term1 + term2 + term3
    return power * delta_t

class UAVISACEnvironment(gym.Env):      # 继承父类gym.Env:任何自定义的Gymnasium环境都必须继承自gym.Env
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14,delta_t: float = 4.0,
                 E_tot: float = 600000.0, energy_penalty: float = 10.0):
        super(UAVISACEnvironment, self).__init__()      
        # super返回的代理对象调用UAVISACEnviroment的父类的__init__函数

        # 时间设置
        self.N = N  # 时隙数量
        self.K = K  # 通信用户数量
        self.H = H  # 无人机固定高度
        self.H1 = H1  # 目标固定高度
        self.sigma2 = sigma2  # 噪声功率
        self.l_max = l_max  # 每个时隙内无人机最大飞行距离

        self.delta_t = delta_t  # 时隙长度固定为 4 s
        self.E_tot = E_tot  # 无人机电池容量 (J)
        self.energy_penalty = energy_penalty  # 超额能耗惩罚权重

        # 无人机飞行范围约束
        self.X_min = -400.0
        self.X_max = 400.0
        self.Y_min = -400.0
        self.Y_max = 400.0

        # 最大发射功率
        self.P_max = 0.1  # 根据需要调整

        # 动作空间：无人机移动角度、距离和发射功率，范围在 [-1, 1]，连续有界，后续解耦成有意义的物理量
        # shape是动作空间向量的维度
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 观察空间：智能体从强化学习环境中接收到的观察信息集合。
        # {无人机位置（x和y）、通信用户位置(x和y)、上一步动作（角度/距离/发射功率共3个量）} * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((2 + 2 * self.K + 3) * 2,), dtype=np.float32
        )
        self.total_energy = 0.0  # 累计能耗 (J)
        self.t1 = 0
        self.rresult = 0
        self.current_episode_reward = 0
        self.episode_rewards = []

        # 初始化观察缓存
        self.prev_obs = None

        # 初始化环境
        self.reset()

    def reset(self, seed=None, options=None):
        # 初始化当前时隙
        super().reset(seed=seed)  # 关键修改！

        # 使用seed初始化随机数生成器
        if seed is not None:
            np.random.seed(seed)  # 设置 Gym 环境中的随机数生成器

        self.current_slot = 0
        self.current_episode_reward = 0
        self.total_energy = 0.0

        x1 = np.random.uniform(-400, 400)
        y1 = np.random.uniform(-400, 400)
        # 初始化无人机位置 (X1, Y1, H)
        self.uav_position = np.array([x1, y1, self.H])

        # 初始化通信用户位置，随机分布在指定范围内
        self.user_positions = []
        for _ in range(self.K):
            # 生成用户位置，避免在 -100 到 100 之间
            while True:
                user_x = np.random.uniform(-400, 400)
                user_y = np.random.uniform(-400, 400)
                if (user_x ** 2 + user_y ** 2) > 10000:  # 50^2=2500
                    break
            self.user_positions.append([user_x, user_y, 0])  # z=0



        self.user_positions = np.array(self.user_positions)  # z=0，列表转为np数组,运算性能优于python原生的数组

        # 初始化目标位置 (x1, y1, H1)，固定不变
        self.target_position = np.array([100.0, 100.0, self.H1])

        # 初始化合法雷达接收器位置 (x0, y0, 0)
        self.radar_receiver_position = np.array([0.0, 0.0, 0.0])  # 可根据需要调整

        # 初始化上一步动作为零
        self.prev_action = np.array([0.0, 0.0, 0.0])

        # 获取当前观察值
        current_obs = self._get_obs()

        # 初始化前一个观察值为全零
        self.prev_obs = np.zeros_like(current_obs)

        # 返回组合后的观察值
        return self._get_combined_obs(), {}

    def _get_obs(self):
        # 返回当前观察值（不包含目标位置）
        obs = np.concatenate([
            self.uav_position[:2],
            self.user_positions[:, :2].flatten(),   # 选取user_position数组每一行的0-1列（n行2列数组），flatten()将二维数组变为一维
            # 不包含目标位置 self.target_position[:2],
            self.prev_action
        ])
        return obs

    def _get_combined_obs(self):
        return np.concatenate([self._get_obs(), self.prev_obs])

    def step(self, action):
        # 将动作从 [-1, 1] 映射到实际值
        angle = action[0] * np.pi  # 映射到 [-π, π)
        distance = action[1] * self.l_max  # 映射到 [-l_max, l_max]
        # 原代码（修改前）
        # power_allocation = abs(action[2]) * self.P_max

        # 修改后：使用Sigmoid约束到[0, P_max]
        power_allocation = (action[2] + 1) / 2 * self.P_max  # 映射到[0, P_max]

        # 更新无人机位置
        delta_x = distance * np.cos(angle)
        delta_y = distance * np.sin(angle)
        new_uav_position = self.uav_position.copy()
        new_uav_position[0] += delta_x
        new_uav_position[1] += delta_y

        # 确保无人机位置在允许范围内

        if new_uav_position[0] < self.X_min or new_uav_position[0] > self.X_max or new_uav_position[1] < self.Y_min or \
                new_uav_position[1] > self.Y_max:
            # 越界，惩罚
            reward = -20.0  # 可以根据需要调整惩罚力度
        else:
            # 计算奖励
            reward = self._calculate_reward(new_uav_position, power_allocation)
            # 更新无人机位置
            self.uav_position = new_uav_position

        # ---------- 能耗计算 ----------
        horizontal_speed = abs(distance) / 4.0  # v_u_t = distance / 4 s
        energy_t = calc_energy(horizontal_speed, self.delta_t)
        self.total_energy += energy_t
        if self.total_energy > self.E_tot:
            reward -= self.energy_penalty

        # 计算奖励
        self.current_episode_reward += reward

        # 记录当前动作以供下一个时隙使用
        self.prev_action = action

        # 更新通信用户位置，供下一个时隙使用
        self._update_user_positions()

        # 增加时间步，积累回合平均奖励
        # t1:训练回合数   current_slot：当前训练回合处在第几时间步
        self.current_slot += 1

        done = False
        if self.current_slot == 50:
            done = True
            # 还没有训练500个回合，则将当前回合平均奖励加入总平均奖励rresult
            if self.t1 < 500:
                self.rresult += self.current_episode_reward / 50
                self.t1 += 1
            # 已经训练500回合，则计算这（0-499）共500个回合平均奖励并存入episode_rewards(忽略第500回合的平均奖励)
            else:
                average_rresult = self.rresult / 500
                self.episode_rewards.append(average_rresult)
                self.rresult = 0
                self.t1 = 0

        # 获取当前观察值
        current_obs = self._get_obs()

        # 组合当前观察和前一个观察
        combined_obs = self._get_combined_obs()

        # 更新前一个观察值为当前观察值
        self.prev_obs = current_obs.copy()

        # 返回组合后的观察值和奖励返回标准的 Gymnasium step() 接口格式：
        # combined_obs: 组合后的观察状态（当前+前一时刻）
        # reward: 本时间步的奖励值
        # done: 是否 episode 结束
        # False: truncated 标志（未截断）
        # {}: 额外信息字典（空）        
        return combined_obs, reward, done, False, {}


    def _calculate_reward(self, uav_position, power_allocation):
        # 计算合法接收机的感知信噪比
        eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
        reward = eta_0
        # Communication SNR thresholds (adjust as needed)
        communication_threshold = 10.0  # Communication SNR threshold
        eavesdropper_threshold = 10.0  # Eavesdropper SNR threshold

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
        # 定义奖励函数，鼓励高的合法感知信噪比和通信信噪比，惩罚高的非法感知信噪比

        return reward

    def _calculate_communication_snr(self, distance, power_allocation):
        # 通信模型中的信道增益计算

        # 常数参数
        c1 = 12.081
        c2 = 0.11395
        mu_los = 1.44544
        mu_nlos = 199.526
        fc = 2.4e9  # 载波频率
        c = 3e8  # 光速
        alpha = 2.0  # 路径损耗指数
        K_0 = (4 * np.pi * fc) / c

        # 计算仰角
        d_3d = np.sqrt(self.H**2 + distance**2)
        theta = np.arcsin(self.H / d_3d) * 180 / np.pi  # 角度制

        # 计算LoS概率
        p_los = 1 / (1 + c1 * np.exp(-c2 * (theta - c1)))
        p_nlos = 1 - p_los

        los = mu_los * (K_0 * d_3d) ** alpha
        nlos = mu_nlos * (K_0 * d_3d) ** alpha
        los = max(los, 1e-5)
        nlos = max(nlos, 1e-5)
        los = 10 * np.log10(los)
        nlos = 10 * np.log10(nlos)

        # 计算平均信道增益
        L = p_los * los + p_nlos * nlos
        L = 10 ** (L / 10)
        omega = 1 / L

        # 计算信噪比
        snr = (omega * power_allocation) / self.sigma2
        snr = max(snr, 1e-5)
        snr_db = 10 * np.log10(snr)

        return snr_db

    def _calculate_sensing_snr_legal(self, uav_position, power_allocation):
        # 计算合法接收机的感知信噪比

        # 距离计算
        d_t = np.linalg.norm(uav_position - self.target_position)
        d_r = np.linalg.norm(self.target_position - self.radar_receiver_position)

        # 常数参数
        G_tx = 13  # 发射天线增益
        G_rx = 13  # 接收天线增益
        c = 3e8
        fc = 2.4e9
        lambda_c = c / fc
        sigma = 1.0  # 目标的雷达截面

        # 计算接收功率
        P_r = (power_allocation * 10 ** (G_tx / 10) * 10 ** (G_rx / 10) * lambda_c**2 * sigma) / \
              (((4 * np.pi)**3) * d_t**2 * d_r**2)

        # 计算信噪比
        snr = P_r / self.sigma2
        snr = max(snr, 1e-5)
        snr_db = 10 * np.log10(snr)
        return snr_db

    def _calculate_sensing_snr_eavesdropper(self, uav_position, power_allocation):
        # 计算非法感知者的感知信噪比列表

        eavesdropper_snr_list = []
        d_t = np.linalg.norm(uav_position - self.target_position)

        # 常数参数
        G_tx = 13  # 发射天线增益
        G_rx = 13  # 接收天线增益
        c = 3e8
        fc = 2.4e9
        lambda_c = c / fc
        sigma = 1.0  # 目标的雷达截面

        for k in range(self.K):
            # 计算目标到非法感知者的距离
            d_k_r = np.linalg.norm(self.target_position - self.user_positions[k])

            # 计算接收功率
            P_r_k = (power_allocation *10 ** (G_tx / 10) * 10 ** (G_rx / 10) * lambda_c**2 * sigma) / \
                    (((4 * np.pi)**3) * d_t**2 * d_k_r**2)

            # 计算信噪比
            snr_k = P_r_k / self.sigma2
            snr_k = max(snr_k, 1e-5)
            snr_db_k = 10 * np.log10(snr_k)
            eavesdropper_snr_list.append(snr_db_k)

        return eavesdropper_snr_list

    def _update_user_positions(self):
        # 更新通信用户的位置，为下一时隙做准备
        # 示例：用户以随机方向和速度移动
        for k in range(self.K):
            original_x = self.user_positions[k, 0]
            original_y = self.user_positions[k, 1]
            valid = False

            # 循环直到生成合法的新位置
            while not valid:
                # 生成移动参数（极坐标）
                move_distance = np.random.uniform(0, 50)
                move_angle = np.random.uniform(-np.pi, np.pi)

                # 计算直角坐标系增量
                delta_x = move_distance * np.cos(move_angle)
                delta_y = move_distance * np.sin(move_angle)

                # 计算新坐标
                new_x = original_x + delta_x
                new_y = original_y + delta_y

                # 计算到原点的距离（使用平方比较避免开根号）
                distance_sq = new_x ** 2 + new_y ** 2

                # 验证条件
                coord_in_range = (-400 <= new_x <= 400) and (-400 <= new_y <= 400)
                safe_distance = distance_sq > 10000  # 50^2 = 2500

                valid = coord_in_range and safe_distance

            # 更新合法坐标
            self.user_positions[k, 0] = new_x
            self.user_positions[k, 1] = new_y
    # 移除了 _update_target_position 方法，因为目标是固定的

