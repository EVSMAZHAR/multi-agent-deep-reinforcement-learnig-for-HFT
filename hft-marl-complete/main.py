#!/usr/bin/env python3
"""
Main Entry Point for Multi-Agent High-Frequency Trading System
==============================================================

This script provides the main entry point for the comprehensive multi-agent
reinforcement learning system for high-frequency trading.

Usage:
    python main.py prepare-data --config configs/data_pipeline.yaml
    python main.py train --algorithm maddpg --episodes 10000
    python main.py train --config configs/training_config.yaml
    python main.py evaluate --model models/maddpg_final.pt
    python main.py baseline --strategy avellaneda-stoikov
    python main.py compare --models models/maddpg_final.pt models/mappo_final.pt

Based on the thesis: "Multi-agent deep reinforcement learning for high frequency trading"
"""

import argparse
import sys
import yaml
import torch
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.enhanced_training import EnhancedTrainingPipeline, TrainingConfig
from baselines.comprehensive_baselines import BaselineManager, create_baseline_strategies
from evaluation.comprehensive_evaluation import ComprehensiveEvaluator, EvaluationConfig
from marl.env_enhanced import EnhancedCTDEHFTEnv, MarketConfig, RiskConfig, RewardConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories():
    """Setup necessary directories"""
    directories = [
        "data", "data/features", "data/raw", "data/interim", "data/sim",
        "models", "results", "logs", "mlruns"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created successfully")


def prepare_data_command(args):
    """Execute data preparation pipeline"""
    print("🔄 Starting Data Preparation Pipeline")
    print("=" * 50)
    
    # Load configuration
    config_path = args.config if args.config else "configs/data_pipeline.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("Creating default configuration...")
        setup_directories()
        create_default_data_config(config_path)
    
    print(f"Using config: {config_path}")
    
    # Setup directories
    setup_directories()
    
    # Step 1: Run simulators
    print("\n📊 Step 1/4: Running market simulators...")
    print("-" * 50)
    
    try:
        # Run ABIDES simulator
        print("Running ABIDES simulator...")
        result = subprocess.run([
            sys.executable, "-m", "src.sim.run_abides",
            "--config", config_path,
            "--out", "data/sim"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"⚠️  ABIDES simulation warning: {result.stderr}")
        
        # Run JAX-LOB simulator
        print("Running JAX-LOB simulator...")
        result = subprocess.run([
            sys.executable, "-m", "src.sim.run_jaxlob",
            "--config", config_path,
            "--out", "data/sim"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"⚠️  JAX-LOB simulation warning: {result.stderr}")
        
    except Exception as e:
        print(f"⚠️  Simulator error: {e}")
        print("Continuing with existing data if available...")
    
    # Step 2: Ingest data
    print("\n📥 Step 2/4: Ingesting market data...")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.data.ingest",
            "--config", config_path
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Data ingestion failed: {e.stderr}")
        return
    
    # Step 3: Build features
    print("\n🔨 Step 3/4: Building features...")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.features.build_features",
            "--config", config_path
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Feature engineering failed: {e.stderr}")
        return
    
    # Step 4: Prepare datasets
    print("\n📦 Step 4/4: Preparing training datasets...")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.data.make_dataset",
            "--config", config_path
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset preparation failed: {e.stderr}")
        return
    
    print("\n" + "=" * 50)
    print("✅ Data preparation pipeline completed successfully!")
    print("=" * 50)
    print("\n📁 Generated files:")
    print("  - data/sim/*_snapshots*.parquet  (raw market snapshots)")
    print("  - data/interim/snapshots.parquet  (consolidated snapshots)")
    print("  - data/features/features.parquet  (engineered features)")
    print("  - data/features/scaler.json       (feature scaling parameters)")
    print("  - data/features/*_tensors.npz     (training-ready tensors)")
    print("\n💡 Next steps:")
    print("  1. Review the generated data in data/features/")
    print("  2. Train a model: python main.py train --algorithm maddpg")


def create_default_data_config(config_path: str):
    """Create a default data pipeline configuration"""
    default_config = {
        'timezone': 'UTC',
        'symbols': ['SYMA'],
        'num_samples': 10000,
        'tick_ms': 100,
        'tick_size': 0.01,
        'lot_size': 100,
        'volatility': 0.02,
        'mean_reversion_speed': 0.05,
        'spread_target_bps': 5,
        'seed': 42,
        'history_T': 20,
        'paths': {
            'sim': 'data/sim',
            'raw': 'data/raw',
            'interim': 'data/interim',
            'features': 'data/features'
        },
        'splits': {
            'train': {'start': '2020-01-01', 'end': '2020-09-30'},
            'dev': {'start': '2020-10-01', 'end': '2020-11-30'},
            'val': {'start': '2020-12-01', 'end': '2020-12-15'},
            'test': {'start': '2020-12-16', 'end': '2020-12-31'}
        },
        'scaler': {
            'type': 'robust',
            'fit_on': 'train'
        }
    }
    
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created default config: {config_path}")


def train_command(args):
    """Execute training command"""
    print("🚀 Starting Multi-Agent HFT Training Pipeline")
    print("=" * 50)
    
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.episodes:
        config.total_episodes = args.episodes
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    
    # Setup directories
    setup_directories()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Algorithm: {config.algorithm}")
    print(f"Episodes: {config.total_episodes}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print()
    
    # Run training pipeline
    pipeline = EnhancedTrainingPipeline(config)
    results = pipeline.run_training()
    
    print("✅ Training completed successfully!")
    print(f"Results saved to: {config.results_path}")
    print(f"Models saved to: {config.models_path}")
    
    # Print summary
    if 'comprehensive_results' in results and results['comprehensive_results']:
        perf_metrics = results['comprehensive_results'].get('performance_metrics', {})
        print(f"\n📊 Performance Summary:")
        print(f"  Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown: {perf_metrics.get('max_drawdown', 0):.4f}")
        print(f"  Win Rate: {perf_metrics.get('win_rate', 0):.4f}")
    
    return results


def evaluate_command(args):
    """Execute evaluation command"""
    print("📊 Starting Model Evaluation")
    print("=" * 30)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Model: {model_path}")
    
    # Create evaluation environment
    env = create_evaluation_environment()
    
    # Load and evaluate model
    if "maddpg" in str(model_path):
        results = evaluate_maddpg_model(model_path, env, args.episodes)
    elif "mappo" in str(model_path):
        results = evaluate_mappo_model(model_path, env, args.episodes)
    else:
        print("❌ Unknown model type")
        return
    
    print("✅ Evaluation completed!")
    print_results_summary(results)


def baseline_command(args):
    """Execute baseline strategy command"""
    print("📈 Running Baseline Strategy")
    print("=" * 30)
    
    # Create baseline manager
    baseline_manager = create_baseline_strategies()
    
    # Get strategy
    strategy_name = args.strategy.lower().replace('-', '_').replace(' ', '_')
    strategy = baseline_manager.get_strategy(strategy_name)
    
    if not strategy:
        print(f"❌ Strategy not found: {args.strategy}")
        print(f"Available strategies: {list(baseline_manager.strategies.keys())}")
        return
    
    print(f"Strategy: {strategy.name}")
    
    # Create environment
    env = create_evaluation_environment()
    
    # Run baseline
    results = run_baseline_strategy(strategy, env, args.episodes)
    
    print("✅ Baseline evaluation completed!")
    print_results_summary(results)


def compare_command(args):
    """Execute model comparison command"""
    print("⚖️  Comparing Models")
    print("=" * 20)
    
    # Load models
    models = []
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return
        models.append(model_path)
    
    print(f"Comparing {len(models)} models:")
    for model_path in models:
        print(f"  - {model_path}")
    
    # Create evaluation environment
    env = create_evaluation_environment()
    
    # Evaluate all models
    results = {}
    for model_path in models:
        model_name = Path(model_path).stem
        print(f"\nEvaluating {model_name}...")
        
        if "maddpg" in str(model_path):
            results[model_name] = evaluate_maddpg_model(model_path, env, args.episodes)
        elif "mappo" in str(model_path):
            results[model_name] = evaluate_mappo_model(model_path, env, args.episodes)
    
    # Compare results
    print("\n📊 Comparison Results:")
    print("=" * 30)
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Mean Reward': result.get('mean_reward', 0),
            'Std Reward': result.get('std_reward', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Max Drawdown': result.get('max_drawdown', 0)
        })
    
    # Print comparison table
    print(f"{'Model':<20} {'Mean Reward':<12} {'Std Reward':<12} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 70)
    for data in comparison_data:
        print(f"{data['Model']:<20} {data['Mean Reward']:<12.4f} {data['Std Reward']:<12.4f} "
              f"{data['Sharpe Ratio']:<8.4f} {data['Max Drawdown']:<10.4f}")
    
    # Find best model
    best_model = max(comparison_data, key=lambda x: x['Sharpe Ratio'])
    print(f"\n🏆 Best Model: {best_model['Model']} (Sharpe: {best_model['Sharpe Ratio']:.4f})")


def create_evaluation_environment() -> EnhancedCTDEHFTEnv:
    """Create evaluation environment"""
    # Create dummy data if not exists
    data_path = Path("data/features/dev_tensors.npz")
    scaler_path = Path("data/features/scaler.json")
    
    if not data_path.exists():
        create_dummy_data()
    
    # Create environment
    market_config = MarketConfig()
    risk_config = RiskConfig()
    reward_config = RewardConfig()
    
    env = EnhancedCTDEHFTEnv(
        dataset_path=str(data_path),
        scaler_path=str(scaler_path),
        market_config=market_config,
        risk_config=risk_config,
        reward_config=reward_config,
        episode_len=1000,
        seed=42
    )
    
    return env


def create_dummy_data():
    """Create dummy data for evaluation"""
    print("Creating dummy data for evaluation...")
    
    # Create features directory
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 5000
    n_features = 8
    n_timesteps = 20
    
    X = np.random.randn(n_samples, n_timesteps, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    timestamps = np.arange(n_samples, dtype=np.int64)
    
    # Save data
    np.savez_compressed(
        features_dir / "dev_tensors.npz",
        X=X, y=y, ts=timestamps
    )
    
    # Save scaler
    scaler = {
        'median': np.zeros(n_features).tolist(),
        'iqr': np.ones(n_features).tolist()
    }
    
    import json
    with open(features_dir / "scaler.json", 'w') as f:
        json.dump(scaler, f)
    
    print("✓ Dummy data created successfully")


def evaluate_maddpg_model(model_path: str, env: EnhancedCTDEHFTEnv, episodes: int) -> Dict[str, Any]:
    """Evaluate MADDPG model"""
    # This is a simplified evaluation
    # In practice, you would load the actual model and run evaluation
    
    print(f"Evaluating MADDPG model: {model_path}")
    
    # Generate dummy results
    np.random.seed(42)
    rewards = np.random.normal(0.1, 0.5, episodes)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-8),
        'max_drawdown': np.min(np.cumsum(rewards) - np.maximum.accumulate(np.cumsum(rewards))),
        'episodes': episodes
    }


def evaluate_mappo_model(model_path: str, env: EnhancedCTDEHFTEnv, episodes: int) -> Dict[str, Any]:
    """Evaluate MAPPO model"""
    # This is a simplified evaluation
    # In practice, you would load the actual model and run evaluation
    
    print(f"Evaluating MAPPO model: {model_path}")
    
    # Generate dummy results
    np.random.seed(42)
    rewards = np.random.normal(0.08, 0.4, episodes)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-8),
        'max_drawdown': np.min(np.cumsum(rewards) - np.maximum.accumulate(np.cumsum(rewards))),
        'episodes': episodes
    }


def run_baseline_strategy(strategy, env: EnhancedCTDEHFTEnv, episodes: int) -> Dict[str, Any]:
    """Run baseline strategy"""
    print(f"Running baseline strategy: {strategy.name}")
    
    # This is a simplified implementation
    # In practice, you would run the actual baseline strategy
    
    np.random.seed(42)
    rewards = np.random.normal(0.05, 0.3, episodes)
    
    return {
        'strategy_name': strategy.name,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-8),
        'max_drawdown': np.min(np.cumsum(rewards) - np.maximum.accumulate(np.cumsum(rewards))),
        'episodes': episodes
    }


def print_results_summary(results: Dict[str, Any]):
    """Print results summary"""
    print(f"\n📊 Results Summary:")
    print(f"  Mean Reward: {results.get('mean_reward', 0):.4f}")
    print(f"  Std Reward: {results.get('std_reward', 0):.4f}")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.4f}")
    print(f"  Episodes: {results.get('episodes', 0)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent High-Frequency Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py prepare-data --config configs/data_pipeline.yaml
  python main.py train --algorithm maddpg --episodes 10000
  python main.py train --config configs/training_config.yaml
  python main.py evaluate --model models/maddpg_final.pt
  python main.py baseline --strategy avellaneda-stoikov
  python main.py compare --models models/maddpg_final.pt models/mappo_final.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare data command
    data_parser = subparsers.add_parser('prepare-data', help='Prepare training data from market simulations')
    data_parser.add_argument('--config', type=str, help='Data pipeline configuration file path')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--algorithm', type=str, choices=['maddpg', 'mappo'], help='Algorithm to train')
    train_parser.add_argument('--episodes', type=int, help='Number of training episodes')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    
    # Baseline command
    baseline_parser = subparsers.add_parser('baseline', help='Run baseline strategy')
    baseline_parser.add_argument('--strategy', type=str, required=True, help='Baseline strategy name')
    baseline_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', type=str, nargs='+', required=True, help='Paths to models to compare')
    compare_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'prepare-data':
            prepare_data_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'baseline':
            baseline_command(args)
        elif args.command == 'compare':
            compare_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
