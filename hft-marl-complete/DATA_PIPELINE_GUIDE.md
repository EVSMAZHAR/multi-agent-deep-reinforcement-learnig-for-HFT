# Data Collection and Feature Engineering Pipeline

## Overview

This document describes the integrated data collection and feature engineering pipeline for the HFT MARL (High-Frequency Trading Multi-Agent Reinforcement Learning) project.

The pipeline consists of five main stages:
1. **Market Simulation** - Generate synthetic market data
2. **Data Ingestion** - Combine and clean simulator outputs
3. **Feature Engineering** - Compute trading features from raw data
4. **Dataset Preparation** - Create training/validation/test splits with scaling
5. **Training** - Use prepared data for RL training

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline Flow                        │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐
│   ABIDES     │     │   JAX-LOB    │
│  Simulator   │     │  Simulator   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                ↓
        ┌───────────────┐
        │ Data Ingestion│
        │  (ingest.py)  │
        └───────┬───────┘
                ↓
     ┌─────────────────────┐
     │ Feature Engineering │
     │(feature_engineering)│
     └─────────┬───────────┘
               ↓
     ┌──────────────────────┐
     │ Dataset Preparation  │
     │  (make_dataset.py)   │
     └──────────┬───────────┘
                ↓
     ┌──────────────────────┐
     │  Training Tensors    │
     │ dev/val/test.npz     │
     └──────────────────────┘
```

## Components

### 1. Market Simulators

#### ABIDES Simulator (`src/simulation/run_abides.py`)
- Generates realistic market microstructure data
- Simulates order book dynamics with realistic price movements
- Includes intraday volatility patterns
- Generates volume clustering events

**Output**: `data/sim/{SYMBOL}_snapshots_abides.parquet`

#### JAX-LOB Simulator (`src/simulation/run_jaxlob.py`)
- Generates alternative market dynamics using JAX-LOB approach
- Multiple frequency components for realistic patterns
- Includes microstructure effects (quote stuffing)
- Different price dynamics from ABIDES for diversity

**Output**: `data/sim/{SYMBOL}_snapshots_jaxlob.parquet`

### 2. Data Ingestion (`src/data/ingest.py`)

Combines multiple simulator outputs into a unified dataset:
- Loads all parquet files from simulation directory
- Sorts by symbol and timestamp
- Removes duplicates
- Validates data quality (positive prices, valid spreads)
- Adds basic derived features (mid price, spread, spread_bps)

**Input**: `data/sim/*.parquet`  
**Output**: `data/interim/snapshots.parquet`

### 3. Feature Engineering (`src/features/feature_engineering.py`)

Computes comprehensive feature set for HFT:

#### Order Book Features
- `spread`: Bid-ask spread
- `spread_bps`: Spread in basis points
- `spread_rel`: Relative spread (normalized)
- `imbalance`: Volume imbalance at best levels
- `depth_imbalance`: Depth-weighted imbalance
- `microprice`: Volume-weighted microprice
- `mid_price`: Mid price

#### Return Features
- `log_return_1`, `log_return_5`, `log_return_10`: Log returns at various windows
- `return_1`, `return_5`, `return_10`: Simple returns

#### Volatility Features
- `realized_vol_10`, `realized_vol_20`, `realized_vol_50`: Realized volatility
- `hl_vol_10`, `hl_vol_20`, `hl_vol_50`: High-low volatility estimator

#### Microstructure Features
- `ofi`: Order flow imbalance
- `ofi_rolling`: Rolling order flow imbalance
- `trade_intensity`: Trading intensity
- `cancel_intensity`: Order cancellation intensity

#### Technical Indicators
- `sma_10`, `sma_20`, `sma_50`: Simple moving averages
- `ema_10`, `ema_20`: Exponential moving averages
- `rsi`: RSI-like momentum indicator

#### Temporal Features
- `hour_sin`, `hour_cos`: Cyclical hour encoding
- `minute_sin`, `minute_cos`: Cyclical minute encoding
- `time_since_open`: Minutes since market open

**Input**: `data/interim/snapshots.parquet`  
**Output**: `data/features/features.parquet`

### 4. Dataset Preparation (`src/data/make_dataset.py`)

Prepares final training datasets:

#### Data Splitting
- Splits data by date ranges (dev/val/test)
- Configurable split boundaries in `data_config.yaml`

#### Feature Scaling
- Supports multiple scaling methods:
  - `robust`: RobustScaler (default, handles outliers well)
  - `standard`: StandardScaler (zero mean, unit variance)
  - `minmax`: MinMaxScaler (scales to [0, 1])
- Fits scaler on dev/train split only
- Applies same scaling to val/test splits

#### Sequence Creation
- Creates temporal sequences of length `history_T` (default: 20)
- Output shape: `[N, T, F]` where:
  - N = number of samples
  - T = history window length
  - F = number of features

#### Target Variables
- Creates forward-looking targets for prediction:
  - `target_1`: 1-step ahead return
  - `target_5`: 5-step ahead return
  - `target_10`: 10-step ahead return

**Input**: `data/features/features.parquet`  
**Output**: 
- `data/features/dev_tensors.npz`
- `data/features/val_tensors.npz`
- `data/features/test_tensors.npz`
- `data/features/scaler.json`

## Configuration

The pipeline is configured via `configs/data_config.yaml`:

```yaml
# Data configuration
symbols: [SYMA, SYMB]
decision_ms: 100
tick_ms: 100

# Paths
paths:
  sim: "data/sim"
  interim: "data/interim"
  features: "data/features"

# Splits
splits:
  dev:
    start: "2025-01-01"
    end: "2025-01-08"
  val:
    start: "2025-01-08"
    end: "2025-01-10"
  test:
    start: "2025-01-10"
    end: "2025-01-13"

# Feature engineering
features:
  history_T: 20
  K_levels: 10

# Scaling
scaling:
  method: robust
  by: symbol_regime
```

## Usage

### Method 1: Using the Bash Script

```bash
cd hft-marl-complete
./scripts/prepare_data.sh
```

### Method 2: Using the Python Script

```bash
cd hft-marl-complete
python scripts/run_pipeline.py --config configs/data_config.yaml --seed 42
```

Options:
- `--config`: Path to data configuration file (default: configs/data_config.yaml)
- `--seed`: Random seed for reproducibility (default: 42)
- `--skip-simulation`: Skip simulation steps and use existing data

### Method 3: Individual Steps

Run each step manually:

```bash
# Step 1: ABIDES simulation
python src/simulation/run_abides.py \
    --config configs/data_config.yaml \
    --out data/sim \
    --seed 42

# Step 2: JAX-LOB simulation
python src/simulation/run_jaxlob.py \
    --config configs/data_config.yaml \
    --out data/sim \
    --seed 42

# Step 3: Data ingestion
python src/data/ingest.py \
    --config configs/data_config.yaml

# Step 4: Feature engineering
python src/features/feature_engineering.py \
    --config configs/data_config.yaml

# Step 5: Dataset preparation
python src/data/make_dataset.py \
    --config configs/data_config.yaml
```

### Method 4: Automatic via Training Pipeline

The training pipeline automatically runs data preparation if needed:

```bash
python main.py --config configs/training_config.yaml
```

## Integration with Training

The prepared datasets are automatically loaded by the training pipeline:

1. `DataManager` checks for existing features
2. If missing, runs the complete pipeline
3. `EnvironmentManager` loads the prepared tensors
4. `EnhancedCTDEHFTEnv` uses the features for observations

### Environment Observations

The environment provides observations with shape `(F + 10,)` where:
- `F`: Number of engineered features (28 by default)
- `+10`: Additional agent and market state features:
  - spread
  - imbalance
  - mid_price
  - best_bid
  - best_ask
  - bid_qty
  - ask_qty
  - inventory
  - cash
  - num_orders

## Output Data Format

### Tensor Files (.npz)

Each split file contains:
- `X`: Feature sequences, shape `[N, T, F]`
  - N: number of samples
  - T: history window (20 by default)
  - F: number of features (28 by default)
- `ts`: Timestamps for each sample
- `symbols`: Symbol identifier for each sample
- `target_1`, `target_5`, `target_10`: Forward-looking returns

### Scaler File (scaler.json)

Contains normalization parameters:
```json
{
  "type": "robust",
  "center": [...],  // Median values
  "scale": [...],   // IQR values
  "feature_names": [...]  // Feature names
}
```

## Data Pipeline Checklist

When setting up the data pipeline:

- [ ] Configure `configs/data_config.yaml` with desired settings
- [ ] Set appropriate date ranges for dev/val/test splits
- [ ] Choose scaling method (robust, standard, or minmax)
- [ ] Set history window length (`history_T`)
- [ ] Run the complete pipeline
- [ ] Verify output files exist in `data/features/`
- [ ] Check tensor shapes match expected dimensions
- [ ] Validate scaler parameters are reasonable
- [ ] Test environment can load the data

## Troubleshooting

### "No simulator files found"
- Ensure simulators ran successfully
- Check `data/sim/` directory for .parquet files
- Verify write permissions on data directory

### "Missing feature files"
- Run the complete pipeline from scratch
- Check for errors in feature engineering step
- Verify all intermediate files exist

### "Feature dimension mismatch"
- Check `history_T` in config matches environment expectations
- Verify number of features matches between pipeline and environment
- Update environment observation space if needed

### "Scaling errors"
- Ensure dev split has enough data
- Check for NaN or infinite values in features
- Verify feature ranges are reasonable

## Extension Guide

### Adding New Features

1. Add feature computation in `src/features/feature_engineering.py`
2. Add feature name to `select_final_features()` function
3. Update feature count in environment if needed
4. Rerun pipeline to regenerate datasets

### Adding New Simulators

1. Create new simulator script in `src/simulation/`
2. Follow same output format (parquet with required columns)
3. Add simulator call to pipeline scripts
4. Update documentation

### Modifying Splits

1. Edit `splits` section in `configs/data_config.yaml`
2. Ensure dates cover available simulated data
3. Rerun dataset preparation step

## Performance Considerations

- **Simulation**: ~1-2 minutes for 10K snapshots per symbol
- **Ingestion**: ~10 seconds for typical dataset
- **Feature Engineering**: ~30 seconds with all features
- **Dataset Preparation**: ~20 seconds with scaling
- **Total Pipeline**: ~3-5 minutes end-to-end

## Best Practices

1. **Always set a seed** for reproducibility
2. **Keep dev split large** (60-70%) for robust scaler fitting
3. **Use robust scaling** for financial data (handles outliers)
4. **Verify data quality** after each pipeline step
5. **Version control configs** for experiment tracking
6. **Monitor feature distributions** to detect issues
7. **Test on small data** before full runs
8. **Document custom features** for future reference

## References

- ABIDES: Agent-Based Interactive Discrete Event Simulation
- JAX-LOB: JAX-based Limit Order Book simulation
- Robust Scaling: Scikit-learn RobustScaler documentation
- Feature Engineering for HFT: Cartea et al. (2015)
