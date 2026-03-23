"""
Unified CSV logging utility for QVPO and SAC experiments.
Provides consistent logging format for training metrics and final comparison.
"""

import csv
import os
from typing import Dict, Any, Optional
import numpy as np


class CSVExperimentLogger:
    """
    Unified CSV logger for RL experiments.
    Ensures consistent field names and formats across different algorithms.
    """
    
    # Define standard field names for training metrics CSV
    TRAINING_METRICS_FIELDS = [
        'run_id',
        'algorithm',
        'seed',
        'scenario_name',
        'eval_interval',
        'step',
        'eval_episode_count',
        'eval_reward_mean',
        'eval_reward_ep_std',
        'eval_leakage_rate',
        'eval_legal_snr_db_mean',
        'eval_legal_snr_db_ep_std',
        'eval_eav_snr_max_db_mean',
        'eval_eav_snr_avg_db_mean',
        'eval_snr_gap_db_mean',
        'train_reward',
        'train_reward_ma100',
        'time_elapsed_sec',
    ]
    
    # Define standard field names for final comparison CSV
    FINAL_COMPARISON_FIELDS = [
        'run_id',
        'algorithm',
        'seed',
        'scenario_name',
        'total_train_steps',
        'eval_interval',
        'eval_episode_count',
        'final_eval_reward',
        'final_eval_reward_ep_std',
        'final_leakage_rate',
        'final_legal_snr_db',
        'final_legal_snr_db_std',
        'final_eav_snr_max_db',
        'final_eav_snr_avg_db',
        'final_snr_gap_db',
        'best_eval_reward',
        'best_step',
        'training_time_sec',
    ]
    
    def __init__(self, run_id: str, algorithm: str, seed: int, 
                 scenario_name: str, eval_interval: int, csv_dir: str):
        """
        Initialize CSV logger.
        
        Args:
            run_id: Unique identifier for this run
            algorithm: Algorithm name (e.g., "QVPO", "SAC")
            seed: Random seed
            scenario_name: Scenario/configuration name
            eval_interval: Evaluation interval in steps
            csv_dir: Directory to save CSV files
        """
        self.run_id = run_id
        self.algorithm = algorithm
        self.seed = seed
        self.scenario_name = scenario_name
        self.eval_interval = eval_interval
        self.csv_dir = csv_dir
        
        # Create CSV directory
        os.makedirs(csv_dir, exist_ok=True)
        
        # CSV file paths
        self.training_metrics_path = os.path.join(
            csv_dir, f'training_metrics_{run_id}.csv'
        )
        self.final_comparison_path = os.path.join(
            csv_dir, f'final_comparison_{run_id}.csv'
        )
        
        # Track logged steps to avoid duplicates
        self.logged_steps = set()
        
        # Initialize training metrics CSV with header
        self._init_training_metrics_csv()
    
    def _init_training_metrics_csv(self):
        """Initialize training metrics CSV file with header."""
        with open(self.training_metrics_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.TRAINING_METRICS_FIELDS)
            writer.writeheader()
    
    def log_training_metrics(self, eval_results: Dict[str, Any], 
                            step: int, time_elapsed_sec: float,
                            train_reward: Optional[float] = None,
                            train_reward_ma100: Optional[float] = None):
        """
        Log training metrics for one evaluation point.
        
        Args:
            eval_results: Dictionary containing evaluation results with keys:
                - mean_return
                - std_return
                - eval_leakage_rate
                - legal_snr_db_mean
                - legal_snr_db_std
                - eav_snr_max_db_mean
                - eav_snr_avg_db_mean
                - snr_gap_db_mean
            step: Current training step
            time_elapsed_sec: Time elapsed since training start (seconds)
            train_reward: Current training reward (optional)
            train_reward_ma100: Training reward MA100 (optional)
        """
        # Check for duplicate step
        if step in self.logged_steps:
            print(f"Warning: Step {step} already logged for {self.algorithm} seed={self.seed}. Skipping.")
            return
        
        # Consistency check: step should be multiple of eval_interval
        if step % self.eval_interval != 0 and step != 0:
            print(f"Warning: Step {step} is not a multiple of eval_interval {self.eval_interval}")
        
        # Prepare row data
        row = {
            'run_id': self.run_id,
            'algorithm': self.algorithm,
            'seed': self.seed,
            'scenario_name': self.scenario_name,
            'eval_interval': self.eval_interval,
            'step': step,
            'eval_episode_count': eval_results.get('eval_episode_count', np.nan),
            'eval_reward_mean': eval_results.get('mean_return', np.nan),
            'eval_reward_ep_std': eval_results.get('std_return', np.nan),
            'eval_leakage_rate': eval_results.get('eval_leakage_rate', np.nan),
            'eval_legal_snr_db_mean': eval_results.get('legal_snr_db_mean', np.nan),
            'eval_legal_snr_db_ep_std': eval_results.get('legal_snr_db_std', np.nan),
            'eval_eav_snr_max_db_mean': eval_results.get('eav_snr_max_db_mean', np.nan),
            'eval_eav_snr_avg_db_mean': eval_results.get('eav_snr_avg_db_mean', np.nan),
            'eval_snr_gap_db_mean': eval_results.get('snr_gap_db_mean', np.nan),
            'train_reward': train_reward if train_reward is not None else np.nan,
            'train_reward_ma100': train_reward_ma100 if train_reward_ma100 is not None else np.nan,
            'time_elapsed_sec': time_elapsed_sec,
        }
        
        # Write to CSV
        with open(self.training_metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.TRAINING_METRICS_FIELDS)
            writer.writerow(row)
        
        # Mark step as logged
        self.logged_steps.add(step)
    
    def log_final_comparison(self, final_eval_results: Dict[str, Any],
                            total_train_steps: int,
                            best_eval_reward: float,
                            best_step: int,
                            training_time_sec: float):
        """
        Log final comparison metrics after training completes.
        
        Args:
            final_eval_results: Dictionary containing final evaluation results with keys:
                - mean_return
                - std_return
                - eval_leakage_rate
                - legal_snr_db_mean
                - legal_snr_db_std
                - eav_snr_max_db_mean
                - eav_snr_avg_db_mean
                - snr_gap_db_mean
                - eval_episode_count
            total_train_steps: Total training steps completed
            best_eval_reward: Best evaluation reward during training
            best_step: Step at which best_eval_reward was achieved
            training_time_sec: Total training time (seconds)
        """
        # Consistency check
        if total_train_steps < max(self.logged_steps) if self.logged_steps else 0:
            print(f"Warning: total_train_steps {total_train_steps} < max logged step {max(self.logged_steps)}")
        
        # Prepare row data
        row = {
            'run_id': self.run_id,
            'algorithm': self.algorithm,
            'seed': self.seed,
            'scenario_name': self.scenario_name,
            'total_train_steps': total_train_steps,
            'eval_interval': self.eval_interval,
            'eval_episode_count': final_eval_results.get('eval_episode_count', np.nan),
            'final_eval_reward': final_eval_results.get('mean_return', np.nan),
            'final_eval_reward_ep_std': final_eval_results.get('std_return', np.nan),
            'final_leakage_rate': final_eval_results.get('eval_leakage_rate', np.nan),
            'final_legal_snr_db': final_eval_results.get('legal_snr_db_mean', np.nan),
            'final_legal_snr_db_std': final_eval_results.get('legal_snr_db_std', np.nan),
            'final_eav_snr_max_db': final_eval_results.get('eav_snr_max_db_mean', np.nan),
            'final_eav_snr_avg_db': final_eval_results.get('eav_snr_avg_db_mean', np.nan),
            'final_snr_gap_db': final_eval_results.get('snr_gap_db_mean', np.nan),
            'best_eval_reward': best_eval_reward,
            'best_step': best_step,
            'training_time_sec': training_time_sec,
        }
        
        # Write to CSV (create with header if doesn't exist)
        file_exists = os.path.isfile(self.final_comparison_path)
        with open(self.final_comparison_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FINAL_COMPARISON_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def create_scenario_name(args) -> str:
    """
    Create a stable scenario name from configuration parameters.
    
    Args:
        args: Argument namespace
        
    Returns:
        Scenario name string
        
    Note:
        - seed is NOT included (recorded separately in CSV)
        - Only key parameters that affect experiment results are included
        - If a parameter doesn't exist, it's skipped (no error)
    """
    # Extract key parameters that define the scenario
    eav_threshold = getattr(args, 'eav_threshold', 10.0)
    comm_threshold = getattr(args, 'comm_threshold', 10.0)
    reward_scale = getattr(args, 'reward_scale', 0.1)
    
    # Build base scenario name
    parts = [
        f"eav{eav_threshold:.1f}",
        f"comm{comm_threshold:.1f}",
        f"rs{reward_scale:.2f}"
    ]
    
    # Add penalty coefficients if they exist
    eav_penalty_coef = getattr(args, 'eav_penalty_coef', None)
    if eav_penalty_coef is not None:
        parts.append(f"epc{eav_penalty_coef:.1f}")
    
    comm_penalty_coef = getattr(args, 'comm_penalty_coef', None)
    if comm_penalty_coef is not None:
        parts.append(f"cpc{comm_penalty_coef:.1f}")
    
    # Add user count if exists (check multiple possible names)
    n_cu = getattr(args, 'N_cu', getattr(args, 'n_cu', getattr(args, 'K', None)))
    if n_cu is not None:
        parts.append(f"k{n_cu}")
    
    # Add episode length if exists (check multiple possible names)
    episode_len = getattr(args, 'max_episode_steps', 
                         getattr(args, 'episode_length', 
                                getattr(args, 'T', None)))
    if episode_len is not None:
        parts.append(f"T{episode_len}")
    
    scenario = "_".join(parts)
    return scenario
