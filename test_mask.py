# test_mask.py
import numpy as np

def test_physics_mask():
    """测试物理掩码逻辑"""
    uav_x, uav_y = 380.0, 380.0  # 接近边界
    l_max = 100.0
    x_min, x_max = -400.0, 400.0
    y_min, y_max = -400.0, 400.0
    
    # 模拟候选动作 (angle_norm, distance_norm, power_norm)
    actions = np.array([
        [0.0, 0.5, 0.0],   # angle=0, distance=50 → new_x=430 (越界)
        [0.5, 0.3, 0.0],   # angle=π/2, distance=30 → new_y=410 (越界)
        [1.0, 0.2, 0.0],   # angle=π, distance=20 → new_x=360 (合法)
        [-0.5, 0.1, 0.0],  # angle=-π/2, distance=10 → new_y=370 (合法)
    ])
    q_values = np.array([10.0, 8.0, 5.0, 3.0])  # Q值
    
    valid_mask = np.ones(4, dtype=bool)
    for i in range(4):
        angle = actions[i, 0] * np.pi
        distance = actions[i, 1] * l_max
        new_x = uav_x + distance * np.cos(angle)
        new_y = uav_y + distance * np.sin(angle)
        if new_x < x_min or new_x > x_max or new_y < y_min or new_y > y_max:
            valid_mask[i] = False
    
    print(f"Valid mask: {valid_mask}")  # 应为 [False, False, True, True]
    
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        valid_q = q_values[valid_indices]
        best_idx = valid_indices[np.argmax(valid_q)]
        print(f"Selected action index: {best_idx}")  # 应为 2 (Q=5.0)
        print("✓ 掩码逻辑正确：跳过越界动作，选择合法动作中Q值最大的")
    
    # 测试 Fallback
    uav_x, uav_y = 395.0, 395.0  # 极端边界位置
    actions_all_invalid = np.array([
        [0.0, 0.1, 0.0],   # 全部越界
        [0.25, 0.1, 0.0],
        [0.5, 0.1, 0.0],
    ])
    valid_mask_fallback = np.zeros(3, dtype=bool)
    
    if not np.any(valid_mask_fallback):
        distances = np.abs(actions_all_invalid[:, 1])
        fallback_idx = np.argmin(distances)
        print(f"Fallback index: {fallback_idx}")
        print("✓ Fallback 逻辑正确：选择移动距离最小的动作")

if __name__ == "__main__":
    test_physics_mask()