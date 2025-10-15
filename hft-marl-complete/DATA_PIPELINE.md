# Data Collection and Feature Engineering Pipeline

This document describes the integrated data collection and feature engineering pipeline for the HFT-MARL project. The pipeline has been adapted from `hft-marl-phase0` and enhanced for compatibility with the sophisticated environment and training algorithms in `hft-marl-complete`.

## Overview

The data pipeline consists of three main stages:

1. **Data Ingestion** - Collect and consolidate raw market data
2. **Feature Engineering** - Compute market microstructure features
3. **Dataset Creation** - Create train/validation/test splits with temporal sequences

## Architecture

```
data/
├── sim/                    # Raw simulator outputs (optional)
├── raw/                    # Raw market data (optional)
├── interim/                # Intermediate processed data
│   └── snapshots.parquet  # Consolidated market snapshots
└── features/              # Final features for training
    ├── features.parquet   # Engineered features
    ├── scaler.json        # Feature scaling parameters
    ├── dev_tensors.npz    # Training data
    ├── val_tensors.npz    # Validation data
    └── test_tensors.npz   # Test data
```

## Configuration

### Data Configuration (`configs/data.yaml`)

Controls data collection and splitting:

```yaml
timezone: UTC
symbols: [SYMA, SYMB]
decision_ms: 100
history_T: 20

paths:
  sim: data/sim
  raw: data/raw
  interim: data/interim
  features: data/features

splits:
  dev:
    start: '2020-01-01'
    end: '2021-12-31'
  val:
    start: '2022-01-01'
    end: '2022-06-30'
  test:
    start: '2022-07-01'
    end: '2022-12-31'

scaling:
  method: robust  # or 'standard'
  by: symbol_regime
```

### Features Configuration (`configs/features.yaml`)

Controls feature engineering:

```yaml
K_levels: 10
history_T: 20
decision_ms: 100

windows:
  ofi_ms: 1000
  realised_vol_ms: 2000
  fast: 10
  slow: 30

aux:
  - spread
  - microprice
  - imbalance
  - mid_price
  - returns
  - volatility_fast
  - volatility_slow
  - spread_ma_fast
  - spread_ma_slow

scaler:
  type: robust  # 'robust' or 'standard'
  fit_on: dev
```

## Pipeline Components

### 1. Data Ingestion (`src/data/ingest.py`)

**Purpose**: Consolidate simulator outputs into a unified format.

**Input**: Parquet or CSV files from `data/sim/`
**Output**: `data/interim/snapshots.parquet`

**Usage**:
```bash
python src/data/ingest.py --config configs/data.yaml
```

**Features**:
- Loads multiple simulator output files
- Sorts and deduplicates by symbol and timestamp
- Handles both parquet and CSV formats
- Creates consolidated market snapshots

### 2. Feature Engineering (`src/features/build_features.py`)

**Purpose**: Compute market microstructure and technical features.

**Input**: `data/interim/snapshots.parquet`
**Output**: 
- `data/features/features.parquet` (engineered features)
- `data/features/scaler.json` (scaling parameters)

**Usage**:
```bash
python src/features/build_features.py --config configs/features.yaml
```

**Features Computed**:

**Basic Features**:
- `spread`: Bid-ask spread
- `imbalance`: Orderbook imbalance
- `microprice`: Volume-weighted mid price
- `mid_price`: Simple mid price
- `returns`: Price returns

**Technical Features**:
- `volatility_{fast,slow}`: Rolling volatility
- `spread_ma_{fast,slow}`: Spread moving averages
- `imbalance_ma_{fast,slow}`: Imbalance moving averages
- `volume_ma_{fast,slow}`: Volume moving averages

**Scaling**:
- Robust scaling (median/IQR) - default, better for outliers
- Standard scaling (mean/std) - alternative option

### 3. Dataset Creation (`src/data/make_dataset.py`)

**Purpose**: Create train/val/test splits with temporal sequences.

**Input**: `data/features/features.parquet`
**Output**: `data/features/{dev,val,test}_tensors.npz`

**Usage**:
```bash
python src/data/make_dataset.py --config configs/data.yaml
```

**Output Format**:
Each `.npz` file contains:
- `X`: Feature sequences [N, T, F]
  - N = number of samples
  - T = temporal window (default: 20)
  - F = number of features (8 base + technical features)
- `y`: Target values [N]
- `ts`: Timestamps [N]

## Integration with Training

### Automatic Pipeline Execution

The training pipeline automatically handles data preparation:

```python
from training.enhanced_training import EnhancedTrainingPipeline, TrainingConfig

config = TrainingConfig(
    algorithm="maddpg",
    total_episodes=10000,
    data_path="data",
    features_path="data/features"
)

pipeline = EnhancedTrainingPipeline(config)
results = pipeline.run_training()
```

The `DataManager` class will:
1. Check if features exist
2. If not, run the complete feature engineering pipeline
3. Generate synthetic data if no raw data is available
4. Create all necessary datasets

### Manual Pipeline Execution

For more control, run the pipeline manually:

```bash
# Run complete pipeline
./scripts/prepare_data.sh

# Or run individual steps
python src/data/ingest.py --config configs/data.yaml
python src/features/build_features.py --config configs/features.yaml
python src/data/make_dataset.py --config configs/data.yaml
```

## Environment Compatibility

The pipeline ensures compatibility with the `EnhancedCTDEHFTEnv`:

### Required Features

The environment expects these core features:
- `best_bid`, `best_ask`: Top of book prices
- `bid_qty_1`, `ask_qty_1`: Top of book quantities
- `spread`: Bid-ask spread
- `imbalance`: Orderbook imbalance
- `microprice`: Volume-weighted mid
- `mid_price`: Simple mid price

### Data Format

- **Shape**: `[N, T, F]` where:
  - N = number of samples (variable)
  - T = temporal window (default: 20)
  - F = number of features (8 base + technical)

- **Scaling**: Robust scaling by default (better for financial data)

- **File Format**: NumPy compressed (`.npz`)

### Agent Types

The environment supports multiple agent types:
- **Market Maker**: Posts limit orders, provides liquidity
- **Market Taker**: Submits market orders, consumes liquidity

Features are compatible with both agent types.

## Synthetic Data Generation

When no raw data is available, the system generates synthetic market data:

**Characteristics**:
- 30 days of data at 100ms intervals
- Geometric Brownian motion for prices
- Realistic spread dynamics
- Pareto-distributed volumes
- Proper orderbook structure

**Purpose**:
- Testing and development
- Demonstrating pipeline functionality
- Quick prototyping

**Note**: Replace with real market data for production use.

## Extending the Pipeline

### Adding New Features

1. Edit `src/features/build_features.py`:
```python
def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    df['my_feature'] = ...  # Your calculation
    return df
```

2. Update `configs/features.yaml`:
```yaml
aux:
  - my_feature
```

3. Ensure compatibility with environment:
   - Check feature dimensions match
   - Verify scaling is applied
   - Test with actual training

### Adding New Data Sources

1. Create a new ingestion module in `src/data/`
2. Ensure output matches expected format:
   - Required columns: `ts`, `best_bid`, `best_ask`, `bid_qty_1`, `ask_qty_1`
   - Optional: `symbol`, additional orderbook levels
3. Update `configs/data.yaml` with new paths

### Custom Scaling

Implement custom scalers in `src/features/build_features.py`:

```python
def compute_scaler(df, feature_cols, method='custom'):
    if method == 'custom':
        scaler = {'method': 'custom', 'params': {...}}
        # Your scaling logic
    return scaler
```

## Troubleshooting

### Issue: Missing Features

**Symptom**: `FileNotFoundError: Features file not found`

**Solution**: 
1. Check if raw data exists in `data/sim/` or `data/raw/`
2. Run `./scripts/prepare_data.sh`
3. Or let training auto-generate synthetic data

### Issue: Dimension Mismatch

**Symptom**: `Shape mismatch in environment`

**Solution**:
1. Check `history_T` matches in configs and environment
2. Verify feature count matches environment expectations
3. Ensure all required features are computed

### Issue: Insufficient Data

**Symptom**: `Insufficient data for split`

**Solution**:
1. Reduce `history_T` in config
2. Adjust split dates in `configs/data.yaml`
3. Generate more synthetic data

### Issue: Scaling Problems

**Symptom**: Training diverges or explodes

**Solution**:
1. Verify scaler is computed on training data only
2. Check for NaN or inf values in features
3. Try switching between 'robust' and 'standard' scaling

## Performance Considerations

### Data Size

- **Small**: < 1M samples - Loads entirely in memory
- **Medium**: 1M - 10M samples - Uses chunking
- **Large**: > 10M samples - Consider data streaming

### Processing Speed

- **Ingestion**: ~10K samples/sec (parquet)
- **Feature Engineering**: ~5K samples/sec
- **Dataset Creation**: ~2K samples/sec (with sequences)

### Memory Usage

- **Per sample**: ~100 bytes (8 features × 20 timesteps × 4 bytes)
- **1M samples**: ~100 MB
- **10M samples**: ~1 GB

## Best Practices

1. **Version Control**: Track config files and pipeline changes
2. **Data Validation**: Check data quality before training
3. **Reproducibility**: Use fixed random seeds
4. **Documentation**: Document any custom features
5. **Testing**: Test pipeline with small data samples first
6. **Monitoring**: Log data statistics during pipeline runs
7. **Backups**: Keep backups of processed data

## References

- Original implementation: `hft-marl-phase0`
- Environment docs: See `src/marl/env_enhanced.py`
- Training pipeline: See `src/training/enhanced_training.py`
