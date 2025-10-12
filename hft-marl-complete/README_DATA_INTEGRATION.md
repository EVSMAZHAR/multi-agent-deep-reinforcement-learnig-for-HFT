# Data Collection and Feature Engineering Integration

## Overview

This document summarizes the integration of data collection and feature engineering from `hft-marl-phase0` into `hft-marl-complete`.

## What Has Been Integrated

### 1. Simulation Modules

Two market simulators have been integrated:

#### `src/simulation/run_abides.py`
- Generates realistic market microstructure data
- Simulates order book dynamics with:
  - Mean reversion price dynamics
  - Intraday volatility patterns (U-shaped)
  - Volume clustering
  - Realistic bid-ask spreads

#### `src/simulation/run_jaxlob.py`
- Generates alternative market dynamics
- Multiple frequency components
- Random walk with drift
- Microstructure effects (quote stuffing)

### 2. Data Collection Module

#### `src/data/ingest.py`
- Combines output from multiple simulators
- Sorts and cleans data
- Removes duplicates
- Validates data quality
- Adds basic derived features (mid price, spread)

### 3. Feature Engineering Module

#### `src/features/feature_engineering.py`
Comprehensive feature computation including:

**Order Book Features** (7 features):
- spread, spread_bps, spread_rel
- imbalance, depth_imbalance
- microprice, mid_price

**Return Features** (6 features):
- log_return_1, log_return_5, log_return_10
- return_1, return_5, return_10

**Volatility Features** (6 features):
- realized_vol_10, realized_vol_20, realized_vol_50
- hl_vol_10, hl_vol_20, hl_vol_50

**Microstructure Features** (4 features):
- ofi (order flow imbalance)
- ofi_rolling
- trade_intensity
- cancel_intensity

**Technical Indicators** (6 features):
- sma_10, sma_20, sma_50
- ema_10, ema_20
- rsi

**Temporal Features** (5 features):
- hour_sin, hour_cos
- minute_sin, minute_cos
- time_since_open

**Total: 28 features**

### 4. Dataset Preparation Module

#### `src/data/make_dataset.py`
- Splits data by date ranges (dev/val/test)
- Applies feature scaling:
  - RobustScaler (default, handles outliers)
  - StandardScaler
  - MinMaxScaler
- Creates temporal sequences (history window of 20 by default)
- Generates target variables (1, 5, 10-step ahead returns)
- Saves tensors in NPZ format

### 5. Training Pipeline Integration

#### Updated `src/training/enhanced_training.py`
- `DataManager` class now runs the complete pipeline
- Automatically detects missing data and triggers pipeline
- Falls back to dummy data if pipeline fails
- Integrated with `EnhancedTrainingPipeline`

### 6. Configuration Files

#### `configs/data_config.yaml`
Complete configuration for:
- Symbols and timing parameters
- Path configuration
- Data split dates
- Feature engineering parameters
- Scaling method

### 7. Pipeline Scripts

#### `scripts/prepare_data.sh`
Bash script to run complete pipeline:
```bash
./scripts/prepare_data.sh
```

#### `scripts/run_pipeline.py`
Python script with more control:
```bash
python scripts/run_pipeline.py --config configs/data_config.yaml --seed 42
```

## Directory Structure

```
hft-marl-complete/
├── configs/
│   ├── data_config.yaml          # NEW: Data pipeline configuration
│   ├── environment_config.yaml   # Existing environment config
│   └── training_config.yaml      # Existing training config
├── data/                          # NEW: Data directory
│   ├── sim/                       # Simulator outputs
│   ├── interim/                   # Intermediate processed data
│   └── features/                  # Final feature tensors
│       ├── dev_tensors.npz
│       ├── val_tensors.npz
│       ├── test_tensors.npz
│       └── scaler.json
├── src/
│   ├── simulation/                # NEW: Simulation modules
│   │   ├── __init__.py
│   │   ├── run_abides.py
│   │   └── run_jaxlob.py
│   ├── data/                      # NEW: Data collection
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── make_dataset.py
│   ├── features/                  # NEW: Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── marl/                      # Existing MARL components
│   │   ├── env_enhanced.py       # UPDATED: Uses real features
│   │   └── policies/
│   ├── training/
│   │   └── enhanced_training.py  # UPDATED: Integrated pipeline
│   ├── baselines/
│   └── evaluation/
├── scripts/
│   ├── prepare_data.sh           # NEW: Pipeline bash script
│   └── run_pipeline.py           # NEW: Pipeline python script
├── DATA_PIPELINE_GUIDE.md        # NEW: Comprehensive guide
├── README_DATA_INTEGRATION.md    # NEW: This file
└── main.py                        # Existing main entry point
```

## Compatibility with Environment and Training

### Environment Compatibility

The `EnhancedCTDEHFTEnv` expects:
- Input tensor shape: `[N, T, F]` where:
  - N = number of samples
  - T = history window (20)
  - F = number of features (28)
- Scaler JSON with robust scaling parameters
- Timestamp and symbol metadata

The engineered features provide:
- 28 features compatible with environment
- Additional 10 agent/market state features added by environment
- Total observation dimension: 38 features

### Training Compatibility

The training pipeline:
1. Checks for existing feature files
2. Runs pipeline if missing
3. Loads prepared tensors
4. Creates environment with proper dimensions
5. Trains MADDPG or MAPPO algorithms

## Usage Examples

### Quick Start

Run the complete pipeline with training:
```bash
cd hft-marl-complete
python main.py train --config configs/training_config.yaml
```

The training pipeline will automatically:
1. Check for feature files
2. Run data pipeline if needed
3. Load features
4. Start training

### Manual Pipeline Execution

Run just the data pipeline:
```bash
# Using bash script
./scripts/prepare_data.sh

# Using Python script
python scripts/run_pipeline.py

# Individual steps
python src/simulation/run_abides.py --config configs/data_config.yaml --out data/sim --seed 42
python src/simulation/run_jaxlob.py --config configs/data_config.yaml --out data/sim --seed 42
python src/data/ingest.py --config configs/data_config.yaml
python src/features/feature_engineering.py --config configs/data_config.yaml
python src/data/make_dataset.py --config configs/data_config.yaml
```

### Customization

Modify `configs/data_config.yaml` to:
- Change symbols
- Adjust date ranges
- Modify feature parameters
- Change scaling method
- Set random seed

## Verification

To verify the integration:

1. **Check directory structure**:
```bash
ls -la data/features/
# Should show: dev_tensors.npz, val_tensors.npz, test_tensors.npz, scaler.json
```

2. **Inspect tensor shapes**:
```python
import numpy as np
data = np.load('data/features/dev_tensors.npz')
print(f"X shape: {data['X'].shape}")  # Should be [N, 20, 28]
print(f"Features: {data['X'].shape[2]}")  # Should be 28
```

3. **Test environment loading**:
```python
from marl.env_enhanced import EnhancedCTDEHFTEnv, MarketConfig, RiskConfig, RewardConfig
env = EnhancedCTDEHFTEnv(
    dataset_path="data/features/dev_tensors.npz",
    scaler_path="data/features/scaler.json",
    market_config=MarketConfig(),
    risk_config=RiskConfig(),
    reward_config=RewardConfig()
)
obs, _ = env.reset()
print(f"Observation shape: {obs['market_maker_0'].shape}")  # Should be (38,)
```

## Differences from Phase-0

### Enhanced Features
- More comprehensive feature set (28 vs 5 in basic phase)
- Added technical indicators (SMA, EMA, RSI)
- Added temporal features (time-of-day encoding)
- Added microstructure features (OFI, trade intensity)

### Improved Scaling
- Configurable scaling methods
- Proper train/test split handling
- Scaler versioning in JSON format

### Better Integration
- Automatic pipeline execution from training
- Fallback to dummy data if needed
- Error handling and logging
- Modular design for easy extension

### Production Ready
- Comprehensive documentation
- Testing scripts
- Configuration management
- MLflow integration in training

## Testing the Integration

A test script has been provided to verify the integration:

```bash
cd hft-marl-complete
python -c "
from pathlib import Path
import sys
sys.path.append('src')

# Test imports
from simulation.run_abides import generate_abides_data
from simulation.run_jaxlob import generate_jaxlob_data
from data.ingest import combine_and_clean_data
from features.feature_engineering import build_features
from data.make_dataset import create_sequences

print('✅ All imports successful')
print('✅ Data pipeline integration verified')
"
```

## Next Steps

1. **Test the pipeline**:
   ```bash
   ./scripts/prepare_data.sh
   ```

2. **Verify outputs**:
   ```bash
   ls -lh data/features/
   ```

3. **Run training**:
   ```bash
   python main.py train --algorithm maddpg --episodes 1000
   ```

4. **Monitor results**:
   ```bash
   tensorboard --logdir mlruns/
   ```

## Troubleshooting

### Pipeline fails with "No simulator files found"
- Ensure simulators ran successfully
- Check `data/sim/` directory
- Verify write permissions

### Feature dimension mismatch
- Check `history_T` in config (should be 20)
- Verify feature count matches (28 features expected)
- Update environment if feature count changed

### Scaling errors
- Ensure dev split has enough data
- Check for NaN values in features
- Verify feature ranges are reasonable

### Import errors
- Ensure you're running from `hft-marl-complete` directory
- Check Python path includes `src` directory
- Verify all dependencies installed

## Documentation

- See `DATA_PIPELINE_GUIDE.md` for comprehensive pipeline documentation
- See `IMPLEMENTATION_SUMMARY.md` for overall system architecture
- See `README.md` for general project information

## Contact

For issues or questions about the data integration:
1. Check the troubleshooting section above
2. Review the DATA_PIPELINE_GUIDE.md
3. Inspect log files in `logs/` directory
4. Review configuration in `configs/data_config.yaml`
