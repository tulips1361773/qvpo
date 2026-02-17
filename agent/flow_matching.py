import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from agent.model import Model

class FlowMatchingPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 n_timesteps=10, 
                 device='cuda'):
        super(FlowMatchingPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 不需要修改 Model，它支持 SinusoidalPosEmb
        self.model = Model(state_dim, action_dim) 

        # 初始化 Flow Matcher (使用 Rectified Flow / Optimal Transport 配置)
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

        self.n_timesteps = n_timesteps 

    # ------------------ Training (Loss Calculation) ------------------ #
    def loss(self, action, state, weights=1.0):
        # x1: 真实数据 (Action)
        x1 = action
        
        # 采样时间 t, 噪声 x0
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=self.device)
        x0 = torch.randn_like(x1)
        
        # 线性插值 (OT Path)
        t_expand = t.view(-1, 1)
        xt = t_expand * x1 + (1 - t_expand) * x0
        
        # 目标速度 (Target Velocity) -> 直线导数
        ut = x1 - x0
        
        # 🔥🔥🔥 关键修改：将 t 放大 1000 倍，适配 SinusoidalPosEmb 的敏感度
        # 这样 input 范围从 [0, 1] 变成 [0, 1000]，与以前 Diffusion 的习惯一致
        t_input = t * 1000.0
        
        # 模型预测
        vt = self.model(xt, t_input, state)
        
        # 计算 Loss
        loss = torch.mean((vt - ut) ** 2, dim=-1) # (B,)
        
        if isinstance(weights, torch.Tensor):
            loss = loss * weights.view(-1)
            
        return loss.mean()

    # ------------------ Inference (ODE Solver) ------------------ #
    @torch.no_grad()
    def ode_solve_loop(self, state, shape):
        """
        使用 Euler 方法求解 ODE
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        
        # 定义步长
        dt = 1.0 / self.n_timesteps
        
        for i in range(self.n_timesteps):
            # 当前时间 t (0 -> 1)
            t_val = i / self.n_timesteps
            t_curr = torch.ones(batch_size, device=self.device) * t_val
            
            # 🔥🔥🔥 关键修改：推理时同样放大 1000 倍
            t_input = t_curr * 1000.0
            
            # 预测速度 v
            v_pred = self.model(x, t_input, state)
            
            # Euler 更新: x_{t+1} = x_t + v * dt
            x = x + v_pred * dt
            
        return x

    @torch.no_grad()
    def sample(self, state, eval=False, q_func=None, normal=False):
        if normal:
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.ode_solve_loop(state, shape)
            action.clamp_(-1., 1.)
            return action

        # QVPO 的逻辑
        sample_count = 16 # 或者根据 args 传入
        if eval: sample_count = 32

        raw_batch_size = state.shape[0]
        state_rep = state.repeat(sample_count, 1)
        shape = (state_rep.shape[0], self.action_dim)
        
        # Run ODE Solver
        action_rep = self.ode_solve_loop(state_rep, shape)
        action_rep.clamp_(-1., 1.)
        
        # 评估 Q 值
        q1, q2 = q_func(state_rep, action_rep)
        q = torch.min(q1, q2)
        
        # Reshape & Select Best
        action_rep = action_rep.view(sample_count, raw_batch_size, -1).transpose(0, 1)
        q = q.view(sample_count, raw_batch_size, -1).transpose(0, 1)
        
        action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1, 1, self.action_dim)
        best_action = action_rep.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
        
        return best_action

    @torch.no_grad()
    def sample_n(self, state, eval=False, times=32, chosen=1, q_func=None, origin=None):
        old_state = state
        raw_batch_size = state.shape[0]
        
        state_rep = state.repeat(times, 1)
        shape = (state_rep.shape[0], self.action_dim)
        
        action = self.ode_solve_loop(state_rep, shape)
        action.clamp_(-1., 1.)
        
        q1, q2 = q_func(state_rep, action)
        q = torch.min(q1, q2)
        
        action = action.view(times, raw_batch_size, -1).transpose(0, 1)
        q = q.view(times, raw_batch_size, -1).transpose(0, 1)
        
        mean = q.mean()
        std = q.std()
        v = q.mean(dim=1, keepdim=True)
        
        if chosen == 1:
            _, q_idx = torch.max(q, dim=1, keepdim=True)
            action_idx = q_idx.repeat(1, 1, self.action_dim)
            q_best = q.gather(dim=1, index=q_idx).view(raw_batch_size, 1)
            best_action = action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
            return old_state, best_action, (q_best, v), (mean, std)
        else:
            q_top, q_idx = torch.topk(q, k=chosen, dim=1)
            action_idx = q_idx.repeat(1, 1, self.action_dim)
            
            flat_state = old_state.repeat(chosen, 1).view(chosen, raw_batch_size, -1).transpose(0,1).contiguous().view(raw_batch_size*chosen, -1)
            flat_action = action.gather(dim=1, index=action_idx).view(raw_batch_size*chosen, -1)
            flat_q = q_top.view(raw_batch_size*chosen, 1)
            
            return flat_state, flat_action, (flat_q, v), (mean, std)

    def forward(self, state, eval=False, q_func=None, normal=False):
        return self.sample(state, eval, q_func, normal)