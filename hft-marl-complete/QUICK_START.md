# Quick Start Guide: Data Pipeline Integration

## Overview

This guide will help you quickly get started with the integrated data collection and feature engineering pipeline in `hft-marl-complete`.

## Prerequisites

```bash
cd hft-marl-complete

# Install dependencies
pip install -r requirements.txt

# Verify Python version (3.8+ required)
python --version
```

## 5-Minute Quick Start

### Option 1: Automatic Training (Recommended)

The simplest way - training automatically runs the data pipeline:

```bash
python main.py train --algorithm maddpg --episodes 1000
```

This will:
1. ✅ Check for existing data
2. ✅ Run simulators if needed
3. ✅ Engineer features
4. ✅ Start training

### Option 2: Manual Pipeline Execution

Run the data pipeline separately:

```bash
# Using bash script (Linux/Mac)
chmod +x scripts/prepare_data.sh
./scripts/prepare_data.sh

# Using Python script (All platforms)
python scripts/run_pipeline.py
```

Then start training:

```bash
python main.py train --algorithm maddpg --episodes 1000
```

## What Gets Created

After running the pipeline, you'll have:

```
data/
├── sim/                           # Simulator outputs
│   ├── SYMA_snapshots_abides.parquet
│   ├── SYMA_snapshots_jaxlob.parquet
│   ├── SYMB_snapshots_abides.parquet
│   └── SYMB_snapshots_jaxlob.parquet
├── interim/                       # Processed data
│   └── snapshots.parquet
└── features/                      # Final tensors
    ├── dev_tensors.npz           # Training data
    ├── val_tensors.npz           # Validation data
    ├── test_tensors.npz          # Test data
    └── scaler.json               # Feature scaling params
```

## Verify Installation

```bash
# Test imports
python -c "import sys; sys.path.append('src'); \
from simulation.run_abides import generate_abides_data; \
from features.feature_engineering import build_features; \
print('✅ Installation verified')"

# Check directory structure
ls -la data/features/
```

## Configuration

Edit `configs/data_config.yaml` to customize:

```yaml
# Main settings
symbols: [SYMA, SYMB]          # Trading symbols
decision_ms: 100                # Time resolution
tick_ms: 100                    # Tick size

# Data splits
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

# Feature settings
features:
  history_T: 20                 # History window
  K_levels: 10                  # Order book levels

# Scaling
scaling:
  method: robust                # robust, standard, or minmax
```

## Pipeline Steps Explained

### Step 1: Market Simulation
```bash
python src/simulation/run_abides.py --config configs/data_config.yaml --out data/sim --seed 42
python src/simulation/run_jaxlob.py --config configs/data_config.yaml --out data/sim --seed 42
```
- Generates synthetic market data
- Two different simulators for diversity
- Output: Parquet files with OHLC, order book data

### Step 2: Data Ingestion
```bash
python src/data/ingest.py --config configs/data_config.yaml
```
- Combines simulator outputs
- Cleans and validates data
- Output: Unified snapshots file

### Step 3: Feature Engineering
```bash
python src/features/feature_engineering.py --config configs/data_config.yaml
```
- Computes 28 trading features:
  - Order book features (spread, imbalance, microprice)
  - Returns and volatility
  - Order flow imbalance
  - Technical indicators (SMA, EMA, RSI)
  - Temporal features
- Output: Features parquet file

### Step 4: Dataset Preparation
```bash
python src/data/make_dataset.py --config configs/data_config.yaml
```
- Splits data by date
- Scales features (robust scaling)
- Creates temporal sequences
- Output: NPZ tensor files + scaler

## Training Examples

### MADDPG Training
```bash
python main.py train \
    --algorithm maddpg \
    --episodes 10000 \
    --seed 42 \
    --device cuda
```

### MAPPO Training
```bash
python main.py train \
    --algorithm mappo \
    --episodes 10000 \
    --seed 42 \
    --device cuda
```

### Custom Configuration
```bash
python main.py train --config configs/training_config.yaml
```

## Monitoring Training

### View Logs
```bash
tail -f logs/*.log
```

### MLflow UI
```bash
mlflow ui --port 5000
# Open browser: http://localhost:5000
```

### TensorBoard
```bash
tensorboard --logdir mlruns/
# Open browser: http://localhost:6006
```

## Common Issues

### Issue: "No simulator files found"
**Solution**: Run simulators manually:
```bash
python src/simulation/run_abides.py --config configs/data_config.yaml --out data/sim --seed 42
```

### Issue: "Feature dimension mismatch"
**Solution**: Check feature count matches:
```python
import numpy as np
data = np.load('data/features/dev_tensors.npz')
print(f"Features: {data['X'].shape[2]}")  # Should be 28
```

### Issue: "Import errors"
**Solution**: Add src to Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Next Steps

1. **Run the pipeline**: `./scripts/prepare_data.sh`
2. **Start training**: `python main.py train --algorithm maddpg --episodes 1000`
3. **Monitor progress**: Check logs and MLflow UI
4. **Evaluate model**: `python main.py evaluate --model models/maddpg_final.pt`
5. **Compare with baselines**: `python main.py baseline --strategy avellaneda-stoikov`

## Advanced Usage

### Custom Data Source

Replace simulators with your own data:
1. Save data to `data/sim/` as parquet files
2. Ensure columns: `ts`, `symbol`, `best_bid`, `best_ask`, `bid_qty_1`, `ask_qty_1`
3. Run from Step 2 (ingestion) onwards

### Custom Features

Edit `src/features/feature_engineering.py`:
```python
def compute_custom_features(df):
    df['my_feature'] = ...  # Your computation
    return df

# Add to build_features()
df = compute_custom_features(df)

# Add to select_final_features()
core_features.append('my_feature')
```

### Different Scaling Methods

Edit `configs/data_config.yaml`:
```yaml
scaling:
  method: standard  # or minmax, robust
```

## Documentation

- **Comprehensive Guide**: `DATA_PIPELINE_GUIDE.md`
- **Integration Details**: `README_DATA_INTEGRATION.md`
- **System Architecture**: `IMPLEMENTATION_SUMMARY.md`
- **Main README**: `README.md`

## Help & Support

1. Check documentation files
2. Review log files in `logs/`
3. Inspect configuration in `configs/`
4. Test individual pipeline steps
5. Verify data shapes and types

## Success Checklist

- [ ] Dependencies installed
- [ ] Configuration customized
- [ ] Pipeline executed successfully
- [ ] Tensor files created
- [ ] Training started
- [ ] Logs show no errors
- [ ] MLflow tracking working
- [ ] Models saving correctly

## Performance Tips

1. **Use GPU**: Add `--device cuda` for faster training
2. **Parallel environments**: Set `parallel_envs` in config
3. **Batch size**: Increase for GPU efficiency
4. **Feature selection**: Remove unused features
5. **Data size**: Start small, scale up gradually

## Example Workflow

```bash
# 1. Setup
cd hft-marl-complete
pip install -r requirements.txt

# 2. Customize config
vim configs/data_config.yaml

# 3. Run pipeline
./scripts/prepare_data.sh

# 4. Verify output
ls -lh data/features/

# 5. Start training
python main.py train --algorithm maddpg --episodes 5000

# 6. Monitor
tail -f logs/*.log
mlflow ui

# 7. Evaluate
python main.py evaluate --model models/maddpg_final.pt

# 8. Compare
python main.py compare --models models/maddpg_final.pt models/mappo_final.pt
```

That's it! You're ready to use the integrated data pipeline. 🚀
