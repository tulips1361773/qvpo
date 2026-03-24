"""
Agent-side Observation Normalizer using Welford algorithm.

This normalizer is applied AFTER retrieving states from replay buffer,
not before storing them. This ensures replay buffer contains consistent
fixed-scaled states, avoiding the off-policy contamination issue.
"""

import numpy as np
import torch


class ObsNormalizer:
    """
    Online observation normalizer using Welford's algorithm for running mean/variance.
    
    Key differences from environment-side normalization:
    - Applied on agent-side (after replay sampling, before feeding to networks)
    - Replay buffer stores fixed-scaled states (not normalized states)
    - Statistics updated only during online sampling, not from replay batches
    - Can be frozen after certain steps to stabilize training
    """
    
    def __init__(self, state_dim, epsilon=1e-8, clip_range=5.0, device='cpu'):
        """
        Args:
            state_dim: Dimension of observation space
            epsilon: Small constant for numerical stability
            clip_range: Clip normalized values to [-clip_range, clip_range]
            device: torch device for tensor operations
        """
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.device = device
        
        # Welford algorithm statistics (kept in float64 for precision)
        self.mean = np.zeros(state_dim, dtype=np.float64)
        self.M2 = np.zeros(state_dim, dtype=np.float64)
        self.var = np.ones(state_dim, dtype=np.float64)
        self.count = 0.0
        
        # Training mode control
        self.training = True
        self.frozen = False  # When frozen, stop updating statistics
        
    def update(self, state):
        """Update running statistics using Welford's online algorithm.
        
        Args:
            state: Single observation (numpy array or torch tensor)
        """
        if self.frozen:
            return
            
        # Convert to numpy if needed
        if isinstance(state, torch.Tensor):
            x = state.detach().cpu().numpy()
        else:
            x = np.asarray(state, dtype=np.float64)
            
        # Welford's algorithm
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        if self.count > 1:
            self.var = self.M2 / self.count
    
    def normalize(self, state, update_stats=None):
        """Normalize observation using current statistics.
        
        Args:
            state: Observation (numpy array or torch tensor, single or batch)
            update_stats: Whether to update statistics. If None, uses self.training
            
        Returns:
            Normalized observation (same type as input)
        """
        if update_stats is None:
            update_stats = self.training and not self.frozen
            
        is_tensor = isinstance(state, torch.Tensor)
        is_batch = len(state.shape) > 1 if is_tensor else (state.ndim > 1)
        
        # Convert to numpy for processing
        if is_tensor:
            x = state.detach().cpu().numpy()
        else:
            x = np.asarray(state, dtype=np.float32)
        
        # Update statistics if requested (only for single samples during online sampling)
        if update_stats and not is_batch:
            self.update(x)
        
        # Normalize
        std = np.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        # Convert back to original type
        if is_tensor:
            return torch.FloatTensor(normalized).to(self.device)
        else:
            return normalized.astype(np.float32)
    
    def normalize_batch(self, states):
        """Normalize a batch of states without updating statistics.
        
        This is used when sampling from replay buffer.
        
        Args:
            states: Batch of observations (torch tensor)
            
        Returns:
            Normalized batch (torch tensor)
        """
        # Always disable stats update for batch normalization
        return self.normalize(states, update_stats=False)
    
    def set_training(self, mode):
        """Set training mode."""
        self.training = mode
    
    def freeze(self):
        """Freeze statistics - stop updating mean/var."""
        self.frozen = True
        print(f"[ObsNormalizer] Statistics frozen at count={self.count:.0f}")
        print(f"  Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        print(f"  Std range: [{np.sqrt(self.var.min()):.4f}, {np.sqrt(self.var.max()):.4f}]")
    
    def unfreeze(self):
        """Unfreeze statistics - resume updating."""
        self.frozen = False
        print(f"[ObsNormalizer] Statistics unfrozen")
    
    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'mean': self.mean.copy(),
            'M2': self.M2.copy(),
            'var': self.var.copy(),
            'count': self.count,
            'frozen': self.frozen,
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.mean = state_dict['mean'].copy()
        self.M2 = state_dict['M2'].copy()
        self.var = state_dict['var'].copy()
        self.count = state_dict['count']
        self.frozen = state_dict.get('frozen', False)
        print(f"[ObsNormalizer] Loaded state with count={self.count:.0f}, frozen={self.frozen}")
    
    def get_stats_summary(self):
        """Get summary of current statistics."""
        std = np.sqrt(self.var + self.epsilon)
        return {
            'count': self.count,
            'mean_min': float(self.mean.min()),
            'mean_max': float(self.mean.max()),
            'mean_mean': float(self.mean.mean()),
            'std_min': float(std.min()),
            'std_max': float(std.max()),
            'std_mean': float(std.mean()),
            'frozen': self.frozen,
        }
