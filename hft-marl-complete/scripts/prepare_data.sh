#!/bin/bash
# Data Preparation Pipeline Script
# =================================
# This script runs the complete data collection and feature engineering pipeline

set -e  # Exit on error

echo "===================================="
echo "HFT MARL Data Preparation Pipeline"
echo "===================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if config files exist
DATA_CONFIG="${PROJECT_ROOT}/configs/data.yaml"
FEATURES_CONFIG="${PROJECT_ROOT}/configs/features.yaml"

if [ ! -f "$DATA_CONFIG" ]; then
    echo "ERROR: Data config not found: $DATA_CONFIG"
    exit 1
fi

if [ ! -f "$FEATURES_CONFIG" ]; then
    echo "ERROR: Features config not found: $FEATURES_CONFIG"
    exit 1
fi

echo "Using configurations:"
echo "  - Data config: $DATA_CONFIG"
echo "  - Features config: $FEATURES_CONFIG"
echo ""

# Step 1: Data Ingestion (if raw data exists)
echo "Step 1: Data Ingestion"
echo "----------------------"
if [ -d "${PROJECT_ROOT}/data/sim" ] && [ "$(ls -A ${PROJECT_ROOT}/data/sim 2>/dev/null)" ]; then
    echo "Found simulator data. Running ingestion..."
    python "${PROJECT_ROOT}/src/data/ingest.py" --config "$DATA_CONFIG"
    echo "✓ Data ingestion completed"
else
    echo "⚠ No simulator data found in data/sim/"
    echo "  Synthetic data will be generated during feature engineering"
fi
echo ""

# Step 2: Feature Engineering
echo "Step 2: Feature Engineering"
echo "---------------------------"
if [ -f "${PROJECT_ROOT}/data/interim/snapshots.parquet" ]; then
    echo "Found market snapshots. Building features..."
    python "${PROJECT_ROOT}/src/features/build_features.py" --config "$FEATURES_CONFIG"
    echo "✓ Feature engineering completed"
else
    echo "⚠ No market snapshots found. Skipping feature engineering."
    echo "  Features will be generated when training starts."
fi
echo ""

# Step 3: Dataset Creation
echo "Step 3: Dataset Creation"
echo "------------------------"
if [ -f "${PROJECT_ROOT}/data/features/features.parquet" ]; then
    echo "Found features. Creating train/val/test splits..."
    python "${PROJECT_ROOT}/src/data/make_dataset.py" --config "$DATA_CONFIG"
    echo "✓ Dataset creation completed"
else
    echo "⚠ No features found. Datasets will be created when training starts."
fi
echo ""

# Verify outputs
echo "Verification"
echo "------------"
FEATURES_DIR="${PROJECT_ROOT}/data/features"

if [ -d "$FEATURES_DIR" ]; then
    echo "Features directory contents:"
    ls -lh "$FEATURES_DIR"
    echo ""
    
    # Check for required files
    REQUIRED_FILES=("dev_tensors.npz" "val_tensors.npz" "test_tensors.npz" "scaler.json")
    ALL_PRESENT=true
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "${FEATURES_DIR}/${file}" ]; then
            echo "✓ Found: $file"
        else
            echo "✗ Missing: $file"
            ALL_PRESENT=false
        fi
    done
    
    echo ""
    if [ "$ALL_PRESENT" = true ]; then
        echo "✓✓✓ Data preparation completed successfully!"
        echo ""
        echo "You can now run training with:"
        echo "  python main.py --config configs/training_config.yaml"
    else
        echo "⚠ Some files are missing. Training will generate them automatically."
    fi
else
    echo "⚠ Features directory not found."
    echo "  Data will be generated when training starts."
fi

echo ""
echo "===================================="
echo "Pipeline execution completed"
echo "===================================="
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
>>>>>>> cursor/integrate-data-collection-and-feature-engineering-33e5
