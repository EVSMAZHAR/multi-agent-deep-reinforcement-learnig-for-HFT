#!/usr/bin/env python3
"""
Integration Verification Script
================================

This script verifies that the data pipeline integration is complete and compatible
with the EnhancedCTDEHFTEnv environment.
"""

import sys
from pathlib import Path
import numpy as np
import json

def check_files():
    """Check that all necessary files exist"""
    print("🔍 Checking file structure...")
    
    required_files = [
        "src/sim/run_abides.py",
        "src/sim/run_jaxlob.py",
        "src/data/ingest.py",
        "src/data/make_dataset.py",
        "src/features/build_features.py",
        "configs/data_pipeline.yaml",
        "configs/features.yaml",
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            all_exist = False
    
    return all_exist


def check_generated_data():
    """Check that data has been generated correctly"""
    print("\n📊 Checking generated data...")
    
    # Check features
    features_file = Path("data/features/features.parquet")
    if features_file.exists():
        print(f"  ✓ {features_file}")
    else:
        print(f"  ✗ {features_file} not found (run: python main.py prepare-data)")
        return False
    
    # Check scaler
    scaler_file = Path("data/features/scaler.json")
    if scaler_file.exists():
        with open(scaler_file, 'r') as f:
            scaler = json.load(f)
        print(f"  ✓ {scaler_file} ({len(scaler)} features)")
    else:
        print(f"  ✗ {scaler_file} not found")
        return False
    
    # Check tensors
    tensors_file = Path("data/features/train_tensors.npz")
    if not tensors_file.exists():
        print(f"  ✗ {tensors_file} not found")
        return False
    
    data = np.load(tensors_file)
    X = data['X']
    y = data['y']
    ts = data['ts']
    
    print(f"  ✓ {tensors_file}")
    print(f"    - X shape: {X.shape}")
    print(f"    - y shape: {y.shape}")
    print(f"    - ts shape: {ts.shape}")
    print(f"    - X dtype: {X.dtype}")
    
    return True


def verify_format_compatibility():
    """Verify data format is compatible with environment"""
    print("\n✅ Verifying environment compatibility...")
    
    # Load tensors
    tensors_file = Path("data/features/train_tensors.npz")
    if not tensors_file.exists():
        print("  ✗ No tensors found")
        return False
    
    data = np.load(tensors_file)
    X = data['X']
    
    # Check shape
    if len(X.shape) != 3:
        print(f"  ✗ Wrong shape: expected 3D [N, T, F], got {X.shape}")
        return False
    
    N, T, F = X.shape
    print(f"  ✓ Correct tensor format: [N={N}, T={T}, F={F}]")
    
    # Check dtype
    if X.dtype != np.float32:
        print(f"  ⚠ Warning: dtype is {X.dtype}, expected float32")
    else:
        print(f"  ✓ Correct dtype: float32")
    
    # Check for NaNs
    if np.isnan(X).any():
        print(f"  ⚠ Warning: {np.isnan(X).sum()} NaN values found")
    else:
        print(f"  ✓ No NaN values")
    
    # Check for Infs
    if np.isinf(X).any():
        print(f"  ⚠ Warning: {np.isinf(X).sum()} Inf values found")
    else:
        print(f"  ✓ No Inf values")
    
    print("\n  Environment expects:")
    print(f"    - dataset['X'] shape: [N, T, F]")
    print(f"  Pipeline generates:")
    print(f"    - X shape: [{N}, {T}, {F}] ✓")
    
    return True


def check_feature_compatibility():
    """Check that features match environment expectations"""
    print("\n🔧 Checking feature compatibility...")
    
    scaler_file = Path("data/features/scaler.json")
    if not scaler_file.exists():
        print("  ✗ Scaler not found")
        return False
    
    with open(scaler_file, 'r') as f:
        scaler = json.load(f)
    
    expected_features = [
        'best_bid', 'best_ask', 'bid_qty_1', 'ask_qty_1',
        'spread', 'imbalance', 'microprice', 'mid_price',
        'returns', 'volatility', 'bid_value', 'ask_value'
    ]
    
    print(f"  Expected features: {len(expected_features)}")
    print(f"  Generated features: {len(scaler)}")
    
    if len(scaler) == len(expected_features):
        print("  ✓ Feature count matches")
    else:
        print(f"  ✗ Feature count mismatch")
        return False
    
    missing = []
    for feat in expected_features:
        if feat not in scaler:
            missing.append(feat)
    
    if missing:
        print(f"  ✗ Missing features: {missing}")
        return False
    else:
        print("  ✓ All expected features present")
    
    return True


def print_summary():
    """Print integration summary"""
    print("\n" + "="*60)
    print("📋 INTEGRATION SUMMARY")
    print("="*60)
    
    print("\n✅ Integration Status: COMPLETE")
    
    print("\n📁 Generated Files:")
    print("  - data/sim/*_snapshots*.parquet")
    print("  - data/interim/snapshots.parquet")
    print("  - data/features/features.parquet")
    print("  - data/features/scaler.json")
    print("  - data/features/train_tensors.npz")
    
    print("\n🚀 Next Steps:")
    print("  1. Review the generated data in data/features/")
    print("  2. Customize configs/data_pipeline.yaml if needed")
    print("  3. Train a model: python main.py train --algorithm maddpg")
    
    print("\n📖 Documentation:")
    print("  - README.md (updated with data pipeline section)")
    print("  - DATA_PIPELINE_INTEGRATION.md (detailed integration guide)")
    print("  - configs/data_pipeline.yaml (pipeline configuration)")
    print("  - configs/features.yaml (feature engineering config)")


def main():
    """Main verification routine"""
    print("="*60)
    print("DATA PIPELINE INTEGRATION VERIFICATION")
    print("="*60)
    print()
    
    # Check files
    if not check_files():
        print("\n❌ File structure incomplete!")
        sys.exit(1)
    
    # Check generated data
    if not check_generated_data():
        print("\n⚠️  Data not generated yet. Run: python main.py prepare-data")
        sys.exit(1)
    
    # Verify compatibility
    if not verify_format_compatibility():
        print("\n❌ Format compatibility check failed!")
        sys.exit(1)
    
    # Check features
    if not check_feature_compatibility():
        print("\n❌ Feature compatibility check failed!")
        sys.exit(1)
    
    # Print summary
    print_summary()
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED - INTEGRATION VERIFIED!")
    print("="*60)


if __name__ == "__main__":
    main()
