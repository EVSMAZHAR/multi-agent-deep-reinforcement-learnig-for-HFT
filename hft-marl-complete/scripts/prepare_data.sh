#!/bin/bash
#
# Data Preparation Pipeline for HFT MARL
# =======================================
# This script runs the complete data pipeline from simulation to feature engineering

set -e  # Exit on error

echo "======================================================================"
echo "HFT MARL Data Preparation Pipeline"
echo "======================================================================"
echo ""

# Configuration
CONFIG_FILE="configs/data_config.yaml"
DATA_DIR="data"
SIM_DIR="${DATA_DIR}/sim"
SEED=42

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Create necessary directories
mkdir -p "$SIM_DIR"
mkdir -p "${DATA_DIR}/interim"
mkdir -p "${DATA_DIR}/features"

echo "Step 1/5: Running ABIDES Market Simulator"
echo "----------------------------------------------------------------------"
python src/simulation/run_abides.py \
    --config "$CONFIG_FILE" \
    --out "$SIM_DIR" \
    --seed $SEED

echo ""
echo "Step 2/5: Running JAX-LOB Market Simulator"
echo "----------------------------------------------------------------------"
python src/simulation/run_jaxlob.py \
    --config "$CONFIG_FILE" \
    --out "$SIM_DIR" \
    --seed $SEED

echo ""
echo "Step 3/5: Ingesting and Combining Market Data"
echo "----------------------------------------------------------------------"
python src/data/ingest.py \
    --config "$CONFIG_FILE"

echo ""
echo "Step 4/5: Engineering Features"
echo "----------------------------------------------------------------------"
python src/features/feature_engineering.py \
    --config "$CONFIG_FILE"

echo ""
echo "Step 5/5: Preparing Final Datasets"
echo "----------------------------------------------------------------------"
python src/data/make_dataset.py \
    --config "$CONFIG_FILE"

echo ""
echo "======================================================================"
echo "Data Preparation Complete!"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  - ${DATA_DIR}/features/dev_tensors.npz"
echo "  - ${DATA_DIR}/features/val_tensors.npz"
echo "  - ${DATA_DIR}/features/test_tensors.npz"
echo "  - ${DATA_DIR}/features/scaler.json"
echo ""
echo "You can now run training with:"
echo "  python main.py --config configs/training_config.yaml"
echo ""
