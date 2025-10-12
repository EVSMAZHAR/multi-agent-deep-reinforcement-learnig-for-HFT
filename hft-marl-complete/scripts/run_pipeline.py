#!/usr/bin/env python3
"""
Complete Data Pipeline Runner for HFT MARL
===========================================

This script provides a Python-based alternative to the bash script
for running the complete data preparation pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and handle errors"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete data preparation pipeline for HFT MARL"
    )
    parser.add_argument(
        '--config',
        default='configs/data_config.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--skip-simulation',
        action='store_true',
        help='Skip simulation steps (use existing data)'
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create necessary directories
    data_dir = Path("data")
    sim_dir = data_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("HFT MARL DATA PREPARATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Skip simulation: {args.skip_simulation}")
    
    success = True
    
    # Step 1: Run ABIDES Simulator
    if not args.skip_simulation:
        cmd = [
            sys.executable,
            "src/simulation/run_abides.py",
            "--config", str(config_path),
            "--out", str(sim_dir),
            "--seed", str(args.seed)
        ]
        if not run_command(cmd, "Step 1/5: Running ABIDES Market Simulator"):
            success = False
        
        # Step 2: Run JAX-LOB Simulator
        cmd = [
            sys.executable,
            "src/simulation/run_jaxlob.py",
            "--config", str(config_path),
            "--out", str(sim_dir),
            "--seed", str(args.seed)
        ]
        if not run_command(cmd, "Step 2/5: Running JAX-LOB Market Simulator"):
            success = False
    else:
        logger.info("\n" + "="*70)
        logger.info("SKIPPING SIMULATION STEPS")
        logger.info("="*70)
    
    # Step 3: Data Ingestion
    cmd = [
        sys.executable,
        "src/data/ingest.py",
        "--config", str(config_path)
    ]
    if not run_command(cmd, "Step 3/5: Ingesting and Combining Market Data"):
        success = False
    
    # Step 4: Feature Engineering
    cmd = [
        sys.executable,
        "src/features/feature_engineering.py",
        "--config", str(config_path)
    ]
    if not run_command(cmd, "Step 4/5: Engineering Features"):
        success = False
    
    # Step 5: Dataset Preparation
    cmd = [
        sys.executable,
        "src/data/make_dataset.py",
        "--config", str(config_path)
    ]
    if not run_command(cmd, "Step 5/5: Preparing Final Datasets"):
        success = False
    
    # Summary
    logger.info("\n" + "="*70)
    if success:
        logger.info("DATA PREPARATION COMPLETE!")
        logger.info("="*70)
        logger.info("\nOutput files:")
        logger.info("  - data/features/dev_tensors.npz")
        logger.info("  - data/features/val_tensors.npz")
        logger.info("  - data/features/test_tensors.npz")
        logger.info("  - data/features/scaler.json")
        logger.info("\nYou can now run training with:")
        logger.info("  python main.py --config configs/training_config.yaml")
    else:
        logger.error("DATA PREPARATION FAILED!")
        logger.error("="*70)
        logger.error("\nPlease check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
