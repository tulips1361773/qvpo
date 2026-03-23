# CSV Logging Implementation Summary

## Overview
This document summarizes the unified CSV logging system added to the QVPO reinforcement learning project for experiment tracking and algorithm comparison.

## Files Modified

### 1. **csv_logger.py** (NEW)
- **Purpose**: Unified CSV logging utility for both QVPO and SAC algorithms
- **Location**: `/home/moqianyu_26/sda/qvpo/csv_logger.py`
- **Key Components**:
  - `CSVExperimentLogger` class: Main logging interface
  - `create_scenario_name()` function: Generates consistent scenario names from config

### 2. **main.py** (MODIFIED)
- **Purpose**: QVPO training script
- **Changes**:
  - Added CSV logger imports
  - Enhanced `evaluate()` function to collect SNR statistics and return unified dict
  - Added CSV logger initialization in `main()`
  - Modified evaluation loop to log training metrics to CSV
  - Added final evaluation and final comparison CSV logging
  - Added best_step tracking

### 3. **sac2.py** (MODIFIED)
- **Purpose**: SAC training script
- **Changes**:
  - Added CSV logger imports
  - Created new `evaluate_sac()` function matching QVPO's evaluate structure
  - Added CSV logger initialization
  - Modified evaluation loop to use new evaluate function and log to CSV
  - Added final evaluation and final comparison CSV logging
  - Added best_step tracking

## CSV File Specifications

### Training Metrics CSV (`training_metrics_[run_id].csv`)

**Purpose**: Records evaluation results at each eval_interval during training for plotting training curves.

**Complete Field List** (18 fields):
```
run_id, algorithm, seed, scenario_name, eval_interval, step, eval_episode_count,
eval_reward_mean, eval_reward_ep_std, eval_leakage_rate, eval_legal_snr_db_mean,
eval_legal_snr_db_ep_std, eval_eav_snr_max_db_mean, eval_eav_snr_avg_db_mean,
eval_snr_gap_db_mean, train_reward, train_reward_ma100, time_elapsed_sec
```

**Field Descriptions**:
- `run_id`: Unique identifier for this training run
- `algorithm`: "QVPO" or "SAC"
- `seed`: Random seed used
- `scenario_name`: Scenario configuration identifier
- `eval_interval`: Evaluation frequency in steps
- `step`: Current training step (must be multiple of eval_interval)
- `eval_episode_count`: Number of episodes used for this evaluation
- `eval_reward_mean`: Mean episode return across eval episodes
- `eval_reward_ep_std`: Standard deviation of episode returns
- `eval_leakage_rate`: Sensing leakage rate (fraction of users with SNR > threshold)
- `eval_legal_snr_db_mean`: Mean legal receiver sensing SNR (dB)
- `eval_legal_snr_db_ep_std`: Std of legal receiver sensing SNR across episodes
- `eval_eav_snr_max_db_mean`: Mean of max eavesdropper SNR (dB) - **Currently NaN, needs env support**
- `eval_eav_snr_avg_db_mean`: Mean of avg eavesdropper SNR (dB) - **Currently NaN, needs env support**
- `eval_snr_gap_db_mean`: Security margin (legal - max_eav) - **Currently NaN**
- `train_reward`: Current training episode reward (NaN if not available)
- `train_reward_ma100`: 100-episode moving average of training reward
- `time_elapsed_sec`: Time elapsed since training start (seconds)

**Logging Frequency**: One row per evaluation (every `eval_interval` steps)

### Final Comparison CSV (`final_comparison_[run_id].csv`)

**Purpose**: Records final evaluation results after training completes for plotting final performance bars.

**Complete Field List** (17 fields):
```
run_id, algorithm, seed, scenario_name, total_train_steps, eval_interval,
eval_episode_count, final_eval_reward, final_eval_reward_ep_std, final_leakage_rate,
final_legal_snr_db, final_legal_snr_db_std, final_eav_snr_max_db, final_eav_snr_avg_db,
final_snr_gap_db, best_eval_reward, best_step, training_time_sec
```

**Field Descriptions**:
- `run_id`: Unique identifier for this training run
- `algorithm`: "QVPO" or "SAC"
- `seed`: Random seed used
- `scenario_name`: Scenario configuration identifier
- `total_train_steps`: Total training steps completed
- `eval_interval`: Evaluation frequency used during training
- `eval_episode_count`: Number of episodes used for final evaluation
- `final_eval_reward`: Final evaluation mean return
- `final_eval_reward_ep_std`: Final evaluation return std
- `final_leakage_rate`: Final sensing leakage rate
- `final_legal_snr_db`: Final legal receiver sensing SNR (dB)
- `final_legal_snr_db_std`: Final legal receiver sensing SNR std
- `final_eav_snr_max_db`: Final max eavesdropper SNR (dB) - **Currently NaN**
- `final_eav_snr_avg_db`: Final avg eavesdropper SNR (dB) - **Currently NaN**
- `final_snr_gap_db`: Final security margin - **Currently NaN**
- `best_eval_reward`: Best evaluation reward achieved during training
- `best_step`: Step at which best_eval_reward was achieved
- `training_time_sec`: Total training time (seconds)

**Logging Frequency**: One row per training run (at completion)

## Evaluate Function Return Structure

Both `evaluate()` in main.py and `evaluate_sac()` in sac2.py return identical dict structure:

```python
{
    'mean_return': float,           # Mean episode return
    'std_return': float,            # Std of episode returns
    'eval_leakage_rate': float,     # Leakage rate
    'legal_snr_db_mean': float,     # Mean legal SNR (dB)
    'legal_snr_db_std': float,      # Std legal SNR
    'eav_snr_max_db_mean': float,   # Mean max eav SNR (currently NaN)
    'eav_snr_avg_db_mean': float,   # Mean avg eav SNR (currently NaN)
    'snr_gap_db_mean': float,       # Security margin (currently NaN)
    'eval_episode_count': int,      # Number of eval episodes
}
```

## Consistency Checks Implemented

1. **Duplicate Step Prevention**: CSV logger tracks logged steps and warns if duplicate step is attempted
2. **Step Alignment Check**: Warns if step is not a multiple of eval_interval
3. **Total Steps Validation**: Warns if total_train_steps < max logged step in final comparison
4. **Field Name Consistency**: Both algorithms use identical CSV field names
5. **NaN Handling**: Missing data written as NaN, not 0.0

## Algorithm Behavior Changes

**IMPORTANT**: ✅ **NO algorithm logic was changed**

- Training loops: Unchanged
- Reward calculations: Unchanged
- Environment step/reset: Unchanged
- Network architectures: Unchanged
- Hyperparameters: Unchanged
- TensorBoard logging: Unchanged (CSV is supplementary)

**Only additions made**:
- CSV file writing
- Enhanced evaluate functions to collect and return more statistics
- Time tracking for training duration
- Best result tracking

## Known Limitations & Future Work

### 1. Eavesdropper SNR Statistics (Currently NaN)
**Issue**: The environment (`myenv3.py`) calculates eavesdropper SNR internally but doesn't return it in the `info` dict.

**Current Status**: 
- `eav_snr_max_db_mean`, `eav_snr_avg_db_mean`, `snr_gap_db_mean` are all NaN
- Only `legal_snr_db_mean` (eta_0) is properly collected

**Solution Required**: Modify `myenv3.py` to include eavesdropper SNR list in info dict:
```python
# In _calculate_reward() method, add to info dict:
info['eavesdropper_snr_list'] = eavesdropper_snr_list  # List of SNR values in dB
```

Then update evaluate functions to collect:
```python
# In evaluate loop:
eav_snr_list = eval_info.get('eavesdropper_snr_list', [])
if len(eav_snr_list) > 0:
    ep_eav_snr_max_list.append(max(eav_snr_list))
    ep_eav_snr_avg_list.append(np.mean(eav_snr_list))
```

### 2. Scenario Name Generation
Currently uses basic parameters (eav_threshold, comm_threshold, reward_scale). May need to include more parameters for complete scenario identification.

## Usage Example

### Running QVPO with CSV Logging
```bash
python main.py --seed 0 --num_steps 2500000
```

**Output Location**:
- TensorBoard logs: `record/Env/policy_type=Diffusion/ratio=0.1/seed=0/run_id=<timestamp>/`
- CSV logs: `record/Env/policy_type=Diffusion/ratio=0.1/seed=0/run_id=<timestamp>/csv_logs/`
  - `training_metrics_<run_id>.csv`
  - `final_comparison_<run_id>.csv`

### Running SAC with CSV Logging
```bash
python sac2.py --seed 1 --total-timesteps 2500000
```

**Output Location**:
- TensorBoard logs: `record/sac/sac_uav_<timestamp>/`
- CSV logs: `record/sac/sac_uav_<timestamp>/csv_logs/`
  - `training_metrics_sac_seed1_<timestamp>.csv`
  - `final_comparison_sac_seed1_<timestamp>.csv`

## Multi-Seed Aggregation Example

After running multiple seeds, aggregate results:

```python
import pandas as pd
import glob

# Load all training metrics
files = glob.glob('**/training_metrics_*.csv', recursive=True)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Filter by algorithm and step
qvpo_data = df[(df['algorithm'] == 'QVPO') & (df['step'] == 100000)]
sac_data = df[(df['algorithm'] == 'SAC') & (df['step'] == 100000)]

# Compute mean and std across seeds
qvpo_mean_reward = qvpo_data['eval_reward_mean'].mean()
qvpo_std_reward = qvpo_data['eval_reward_mean'].std()
```

## Verification Checklist

- [x] CSV files created in correct location
- [x] Field names identical between QVPO and SAC
- [x] Seed explicitly included in CSV (not just filename)
- [x] Algorithm name explicitly included
- [x] Scenario name included
- [x] Eval interval recorded
- [x] Step values are multiples of eval_interval
- [x] No duplicate steps in training_metrics
- [x] Final evaluation performed after training
- [x] Best result tracked throughout training
- [x] Training time recorded
- [x] NaN used for missing data (not 0.0)
- [x] No algorithm behavior changed
- [x] TensorBoard logging preserved

## File Locations Summary

```
/home/moqianyu_26/sda/qvpo/
├── csv_logger.py              # NEW - CSV logging utility
├── main.py                    # MODIFIED - QVPO with CSV logging
├── sac2.py                    # MODIFIED - SAC with CSV logging
├── myenv3.py                  # UNCHANGED - Environment (needs future update for eav SNR)
└── CSV_LOGGING_SUMMARY.md     # This file
```

## Next Steps for Complete Implementation

1. **Modify myenv3.py** to return eavesdropper SNR list in info dict
2. **Update evaluate functions** to collect eavesdropper SNR statistics
3. **Test with multiple seeds** to verify CSV aggregation works correctly
4. **Create plotting scripts** to visualize training curves and final comparisons
5. **Document scenario naming convention** if more parameters are needed

---

**Implementation Date**: 2026-03-24  
**Status**: ✅ Core CSV logging complete, ⚠️ Eavesdropper SNR collection pending environment update
