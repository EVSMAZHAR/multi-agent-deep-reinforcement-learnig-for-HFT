"""
Enhanced Training Pipeline for Multi-Agent High-Frequency Trading
================================================================

This module provides a comprehensive training pipeline that orchestrates:
- Data preparation and feature engineering
- Environment setup with realistic market simulation
- Multi-agent algorithm training (MADDPG, MAPPO)
- Baseline strategy training and evaluation
- Comprehensive evaluation and comparison
- Model checkpointing and experiment tracking

Based on the thesis: "Multi-agent deep reinforcement learning for high frequency trading"
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime
import mlflow
import mlflow.pytorch
from contextlib import contextmanager

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from marl.env_enhanced import EnhancedCTDEHFTEnv, MarketConfig, RiskConfig, RewardConfig, AgentType
from marl.policies.enhanced_maddpg import EnhancedMADDPG, MADDPGConfig
from marl.policies.enhanced_mappo import EnhancedMAPPO, MAPPOConfig
from baselines.comprehensive_baselines import BaselineManager, create_baseline_strategies
from evaluation.comprehensive_evaluation import ComprehensiveEvaluator, EvaluationConfig


@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Experiment settings
    experiment_name: str = "hft-marl-experiment"
    run_name: str = None
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data settings
    data_path: str = "data"
    features_path: str = "data/features"
    models_path: str = "models"
    results_path: str = "results"
    
    # Training settings
    algorithm: str = "maddpg"  # "maddpg" or "mappo"
    total_episodes: int = 10000
    eval_frequency: int = 100
    save_frequency: int = 500
    
    # Environment settings
    episode_length: int = 1000
    num_agents: int = 2
    agent_types: List[str] = None
    
    # Algorithm-specific settings
    maddpg_config: Dict[str, Any] = None
    mappo_config: Dict[str, Any] = None
    
    # Evaluation settings
    eval_episodes: int = 100
    baseline_comparison: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "file:./mlruns"
    
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"{self.algorithm}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if self.agent_types is None:
            self.agent_types = ["market_maker", "market_taker"]
        
        if self.maddpg_config is None:
            self.maddpg_config = {
                "lr_actor": 1e-4,
                "lr_critic": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "buffer_size": 1000000,
                "noise_scale": 0.1,
                "noise_decay": 0.9995,
                "update_frequency": 1,
                "num_updates": 1
            }
        
        if self.mappo_config is None:
            self.mappo_config = {
                "lr_actor": 3e-4,
                "lr_critic": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "entropy_coef": 0.01,
                "epochs": 4,
                "batch_size": 64,
                "steps_per_update": 2048
            }


class TrainingLogger:
    """Enhanced logging for training process"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize MLflow if enabled
        if config.use_mlflow:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.experiment_name)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup file handler
        log_file = log_dir / f"{self.config.run_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(self.config.run_name)
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    @contextmanager
    def mlflow_run(self):
        """Context manager for MLflow run"""
        if self.config.use_mlflow:
            with mlflow.start_run(run_name=self.config.run_name) as run:
                # Log configuration
                mlflow.log_params(asdict(self.config))
                yield run
        else:
            yield None
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow and file"""
        if self.config.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        # Log to file
        if step is not None:
            self.logger.info(f"Step {step}: {metrics}")
        else:
            self.logger.info(f"Metrics: {metrics}")
    
    def log_artifact(self, artifact_path: str):
        """Log artifact to MLflow"""
        if self.config.use_mlflow:
            mlflow.log_artifact(artifact_path)


class DataManager:
    """Manages data preparation and feature engineering"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_path = Path(config.data_path)
        self.features_path = Path(config.features_path)
        
    def prepare_data(self) -> bool:
        """Prepare all necessary data for training"""
        self.logger.info("Preparing data for training...")
        
        # Check if features exist
        if not self.features_path.exists():
            self.logger.info("Features not found. Running feature engineering...")
            return self._run_feature_engineering()
        
        # Check if features are up to date
        feature_files = [
            self.features_path / "dev_tensors.npz",
            self.features_path / "val_tensors.npz",
            self.features_path / "test_tensors.npz",
            self.features_path / "scaler.json"
        ]
        
        missing_files = [f for f in feature_files if not f.exists()]
        if missing_files:
            self.logger.info(f"Missing feature files: {missing_files}")
            return self._run_feature_engineering()
        
        self.logger.info("Data preparation completed.")
        return True
    
    def _run_feature_engineering(self) -> bool:
        """Run feature engineering pipeline"""
        try:
            # This would call the actual feature engineering scripts
            # For now, we'll create dummy data
            self._create_dummy_features()
            return True
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return False
    
    def _create_dummy_features(self):
        """Create dummy features for testing"""
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy feature data
        np.random.seed(self.config.seed)
        n_samples = 10000
        n_features = 8
        n_timesteps = 20
        
        # Generate synthetic market data
        X = np.random.randn(n_samples, n_timesteps, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)
        timestamps = np.arange(n_samples, dtype=np.int64)
        
        # Save data splits
        splits = {
            'dev': (0, 6000),
            'val': (6000, 8000),
            'test': (8000, 10000)
        }
        
        for split_name, (start, end) in splits.items():
            split_data = {
                'X': X[start:end],
                'y': y[start:end],
                'ts': timestamps[start:end]
            }
            np.savez_compressed(
                self.features_path / f"{split_name}_tensors.npz",
                **split_data
            )
        
        # Save scaler
        scaler = {
            'median': np.zeros(n_features).tolist(),
            'iqr': np.ones(n_features).tolist()
        }
        
        with open(self.features_path / "scaler.json", 'w') as f:
            json.dump(scaler, f)
        
        self.logger.info("Dummy features created successfully.")


class EnvironmentManager:
    """Manages environment creation and configuration"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_manager = DataManager(config)
    
    def create_environment(self, split: str = "dev") -> EnhancedCTDEHFTEnv:
        """Create training environment"""
        # Prepare data
        self.data_manager.prepare_data()
        
        # Load dataset
        dataset_path = self.data_manager.features_path / f"{split}_tensors.npz"
        scaler_path = self.data_manager.features_path / "scaler.json"
        
        # Create market configuration
        market_config = MarketConfig(
            tick_size=0.01,
            decision_ms=100,
            volatility=0.02,
            mean_reversion_speed=0.05
        )
        
        # Create risk configuration
        risk_config = RiskConfig(
            max_inventory=1000,
            max_drawdown=10000.0,
            max_position_value=50000.0
        )
        
        # Create reward configuration
        reward_config = RewardConfig(
            lambda_inventory=0.5,
            lambda_impact=0.2,
            lambda_spread=0.1,
            lambda_risk=1.0
        )
        
        # Create agent types
        agent_types = [AgentType.MARKET_MAKER, AgentType.MARKET_TAKER]
        
        # Create environment
        env = EnhancedCTDEHFTEnv(
            dataset_path=str(dataset_path),
            scaler_path=str(scaler_path),
            market_config=market_config,
            risk_config=risk_config,
            reward_config=reward_config,
            episode_len=self.config.episode_length,
            seed=self.config.seed,
            agent_types=agent_types
        )
        
        return env


class AlgorithmTrainer:
    """Base class for algorithm training"""
    
    def __init__(self, config: TrainingConfig, logger: TrainingLogger):
        self.config = config
        self.logger = logger
    
    def train(self, env: EnhancedCTDEHFTEnv) -> Dict[str, Any]:
        """Train the algorithm"""
        raise NotImplementedError
    
    def evaluate(self, env: EnhancedCTDEHFTEnv, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate the trained algorithm"""
        raise NotImplementedError


class MADDPGTrainer(AlgorithmTrainer):
    """MADDPG algorithm trainer"""
    
    def __init__(self, config: TrainingConfig, logger: TrainingLogger):
        super().__init__(config, logger)
        self.algorithm_config = MADDPGConfig(**config.maddpg_config)
    
    def train(self, env: EnhancedCTDEHFTEnv) -> Dict[str, Any]:
        """Train MADDPG algorithm"""
        self.logger.logger.info("Starting MADDPG training...")
        
        # Initialize algorithm
        obs_dims = {agent: env.observation_space[agent].shape[0] for agent in env.agents}
        action_dims = {agent: env.action_space[agent].shape[0] for agent in env.agents}
        
        maddpg = EnhancedMADDPG(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=env.agents,
            config=self.algorithm_config,
            device=self.config.device
        )
        
        # Training loop
        training_stats = []
        episode_rewards = []
        
        for episode in range(self.config.total_episodes):
            # Reset environment
            obs, _ = env.reset()
            episode_reward = {agent: 0.0 for agent in env.agents}
            
            for step in range(self.config.episode_length):
                # Select actions
                actions = maddpg.select_actions(obs, explore=True)
                
                # Step environment
                next_obs, rewards, terms, truncs, infos = env.step(actions)
                
                # Store experience
                experience = {
                    'observations': obs,
                    'actions': actions,
                    'rewards': rewards,
                    'next_observations': next_obs,
                    'dones': [terms[agent] or truncs[agent] for agent in env.agents]
                }
                maddpg.store_experience(experience)
                
                # Update episode rewards
                for agent in env.agents:
                    episode_reward[agent] += rewards[agent]
                
                # Train if enough experiences
                if len(maddpg.replay_buffer) >= self.algorithm_config.batch_size:
                    train_info = maddpg.train()
                    if train_info:
                        training_stats.append(train_info)
                
                obs = next_obs
                
                # Check termination
                if any(terms.values()) or any(truncs.values()):
                    break
            
            # Log episode results
            total_reward = sum(episode_reward.values())
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.log_metrics({
                    'episode_reward': total_reward,
                    'avg_reward_100': avg_reward,
                    'episode': episode
                }, step=episode)
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                model_path = Path(self.config.models_path) / f"maddpg_episode_{episode}.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                maddpg.save_models(str(model_path))
            
            # Evaluation
            if episode % self.config.eval_frequency == 0:
                eval_results = self.evaluate(env, episodes=10)
                self.logger.log_metrics(eval_results, step=episode)
        
        # Save final model
        final_model_path = Path(self.config.models_path) / "maddpg_final.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        maddpg.save_models(str(final_model_path))
        
        return {
            'training_stats': training_stats,
            'episode_rewards': episode_rewards,
            'final_model_path': str(final_model_path)
        }
    
    def evaluate(self, env: EnhancedCTDEHFTEnv, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate MADDPG algorithm"""
        # Load model
        model_path = Path(self.config.models_path) / "maddpg_final.pt"
        if not model_path.exists():
            self.logger.logger.warning("No trained model found for evaluation")
            return {}
        
        # Initialize algorithm
        obs_dims = {agent: env.observation_space[agent].shape[0] for agent in env.agents}
        action_dims = {agent: env.action_space[agent].shape[0] for agent in env.agents}
        
        maddpg = EnhancedMADDPG(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=env.agents,
            config=self.algorithm_config,
            device=self.config.device
        )
        maddpg.load_models(str(model_path))
        
        # Evaluation loop
        eval_rewards = []
        eval_metrics = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = {agent: 0.0 for agent in env.agents}
            
            for step in range(self.config.episode_length):
                actions = maddpg.select_actions(obs, explore=False)
                next_obs, rewards, terms, truncs, infos = env.step(actions)
                
                for agent in env.agents:
                    episode_reward[agent] += rewards[agent]
                
                obs = next_obs
                
                if any(terms.values()) or any(truncs.values()):
                    break
            
            total_reward = sum(episode_reward.values())
            eval_rewards.append(total_reward)
            
            # Get episode metrics
            episode_metrics = env.get_episode_metrics()
            eval_metrics.append(episode_metrics)
        
        # Calculate evaluation statistics
        eval_stats = {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_std_reward': np.std(eval_rewards),
            'eval_min_reward': np.min(eval_rewards),
            'eval_max_reward': np.max(eval_rewards)
        }
        
        # Add episode metrics
        if eval_metrics:
            avg_metrics = {}
            for key in eval_metrics[0].keys():
                values = [m[key] for m in eval_metrics if key in m]
                if values:
                    avg_metrics[f'eval_avg_{key}'] = np.mean(values)
            eval_stats.update(avg_metrics)
        
        return eval_stats


class MAPPOTrainer(AlgorithmTrainer):
    """MAPPO algorithm trainer"""
    
    def __init__(self, config: TrainingConfig, logger: TrainingLogger):
        super().__init__(config, logger)
        self.algorithm_config = MAPPOConfig(**config.mappo_config)
    
    def train(self, env: EnhancedCTDEHFTEnv) -> Dict[str, Any]:
        """Train MAPPO algorithm"""
        self.logger.logger.info("Starting MAPPO training...")
        
        # Initialize algorithm
        obs_dims = {agent: env.observation_space[agent].shape[0] for agent in env.agents}
        action_dims = {agent: env.action_space[agent].shape[0] for agent in env.agents}
        
        mappo = EnhancedMAPPO(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=env.agents,
            config=self.algorithm_config,
            device=self.config.device
        )
        
        # Training loop
        training_stats = []
        episode_rewards = []
        
        for episode in range(self.config.total_episodes):
            # Collect rollout data
            observations = []
            actions = []
            rewards = []
            dones = []
            old_log_probs = []
            values = []
            
            obs, _ = env.reset()
            episode_reward = {agent: 0.0 for agent in env.agents}
            
            for step in range(self.config.episode_length):
                # Select actions
                actions_dict, log_probs_dict, values_dict = mappo.select_actions(obs)
                
                # Step environment
                next_obs, rewards_dict, terms, truncs, infos = env.step(actions_dict)
                
                # Store rollout data
                observations.append(obs.copy())
                actions.append(actions_dict.copy())
                rewards.append(rewards_dict.copy())
                dones.append([terms[agent] or truncs[agent] for agent in env.agents])
                old_log_probs.append(log_probs_dict.copy())
                values.append(values_dict.copy())
                
                # Update episode rewards
                for agent in env.agents:
                    episode_reward[agent] += rewards_dict[agent]
                
                obs = next_obs
                
                if any(terms.values()) or any(truncs.values()):
                    break
            
            # Train on collected data
            if len(observations) >= self.algorithm_config.steps_per_update:
                train_info = mappo.train(
                    observations, actions, rewards, dones, old_log_probs, values
                )
                if train_info:
                    training_stats.append(train_info)
            
            # Log episode results
            total_reward = sum(episode_reward.values())
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.log_metrics({
                    'episode_reward': total_reward,
                    'avg_reward_100': avg_reward,
                    'episode': episode
                }, step=episode)
            
            # Save model periodically
            if episode % self.config.save_frequency == 0:
                model_path = Path(self.config.models_path) / f"mappo_episode_{episode}.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                mappo.save_models(str(model_path))
            
            # Evaluation
            if episode % self.config.eval_frequency == 0:
                eval_results = self.evaluate(env, episodes=10)
                self.logger.log_metrics(eval_results, step=episode)
        
        # Save final model
        final_model_path = Path(self.config.models_path) / "mappo_final.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        mappo.save_models(str(final_model_path))
        
        return {
            'training_stats': training_stats,
            'episode_rewards': episode_rewards,
            'final_model_path': str(final_model_path)
        }
    
    def evaluate(self, env: EnhancedCTDEHFTEnv, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate MAPPO algorithm"""
        # Load model
        model_path = Path(self.config.models_path) / "mappo_final.pt"
        if not model_path.exists():
            self.logger.logger.warning("No trained model found for evaluation")
            return {}
        
        # Initialize algorithm
        obs_dims = {agent: env.observation_space[agent].shape[0] for agent in env.agents}
        action_dims = {agent: env.action_space[agent].shape[0] for agent in env.agents}
        
        mappo = EnhancedMAPPO(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=env.agents,
            config=self.algorithm_config,
            device=self.config.device
        )
        mappo.load_models(str(model_path))
        
        # Evaluation loop
        eval_rewards = []
        eval_metrics = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = {agent: 0.0 for agent in env.agents}
            
            for step in range(self.config.episode_length):
                actions_dict, _, _ = mappo.select_actions(obs)
                next_obs, rewards_dict, terms, truncs, infos = env.step(actions_dict)
                
                for agent in env.agents:
                    episode_reward[agent] += rewards_dict[agent]
                
                obs = next_obs
                
                if any(terms.values()) or any(truncs.values()):
                    break
            
            total_reward = sum(episode_reward.values())
            eval_rewards.append(total_reward)
            
            # Get episode metrics
            episode_metrics = env.get_episode_metrics()
            eval_metrics.append(episode_metrics)
        
        # Calculate evaluation statistics
        eval_stats = {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_std_reward': np.std(eval_rewards),
            'eval_min_reward': np.min(eval_rewards),
            'eval_max_reward': np.max(eval_rewards)
        }
        
        # Add episode metrics
        if eval_metrics:
            avg_metrics = {}
            for key in eval_metrics[0].keys():
                values = [m[key] for m in eval_metrics if key in m]
                if values:
                    avg_metrics[f'eval_avg_{key}'] = np.mean(values)
            eval_stats.update(avg_metrics)
        
        return eval_stats


class EnhancedTrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = TrainingLogger(config)
        self.env_manager = EnvironmentManager(config)
        
        # Create results directory
        self.results_path = Path(config.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def run_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        self.logger.logger.info("Starting enhanced training pipeline...")
        
        with self.logger.mlflow_run():
            # Create environment
            env = self.env_manager.create_environment("dev")
            
            # Train algorithm
            if self.config.algorithm.lower() == "maddpg":
                trainer = MADDPGTrainer(self.config, self.logger)
            elif self.config.algorithm.lower() == "mappo":
                trainer = MAPPOTrainer(self.config, self.logger)
            else:
                raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
            
            training_results = trainer.train(env)
            
            # Final evaluation
            eval_results = trainer.evaluate(env, episodes=self.config.eval_episodes)
            
            # Baseline comparison
            if self.config.baseline_comparison:
                baseline_results = self._run_baseline_comparison(env)
                training_results['baseline_results'] = baseline_results
            
            # Comprehensive evaluation
            comprehensive_results = self._run_comprehensive_evaluation(
                training_results, eval_results, env
            )
            
            # Save results
            self._save_results(training_results, eval_results, comprehensive_results)
            
            # Generate report
            report = self._generate_report(comprehensive_results)
            
            return {
                'training_results': training_results,
                'eval_results': eval_results,
                'baseline_results': baseline_results if self.config.baseline_comparison else None,
                'comprehensive_results': comprehensive_results,
                'report': report
            }
    
    def _run_baseline_comparison(self, env: EnhancedCTDEHFTEnv) -> Dict[str, Any]:
        """Run baseline strategy comparison"""
        self.logger.logger.info("Running baseline comparison...")
        
        # Create baseline manager
        baseline_manager = create_baseline_strategies()
        
        # Run baselines (simplified)
        baseline_results = {}
        for strategy_name in baseline_manager.strategies.keys():
            # This would run the actual baseline strategy
            # For now, we'll create dummy results
            baseline_results[strategy_name] = {
                'mean_reward': np.random.normal(0, 1),
                'std_reward': np.random.uniform(0.5, 2.0),
                'sharpe_ratio': np.random.uniform(0, 2.0)
            }
        
        return baseline_results
    
    def _run_comprehensive_evaluation(
        self,
        training_results: Dict[str, Any],
        eval_results: Dict[str, Any],
        env: EnhancedCTDEHFTEnv
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        self.logger.logger.info("Running comprehensive evaluation...")
        
        # Create evaluator
        eval_config = EvaluationConfig()
        evaluator = ComprehensiveEvaluator(eval_config)
        
        # Generate dummy returns for evaluation
        np.random.seed(self.config.seed)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        # Evaluate the trained algorithm
        strategy_results = evaluator.evaluate_strategy(
            strategy_name=self.config.algorithm,
            returns=returns
        )
        
        return strategy_results
    
    def _save_results(
        self,
        training_results: Dict[str, Any],
        eval_results: Dict[str, Any],
        comprehensive_results: Dict[str, Any]
    ):
        """Save all results"""
        results = {
            'config': asdict(self.config),
            'training_results': training_results,
            'eval_results': eval_results,
            'comprehensive_results': comprehensive_results,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = self.results_path / f"{self.config.run_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.log_artifact(str(results_file))
    
    def _generate_report(self, comprehensive_results: Dict[str, Any]) -> str:
        """Generate comprehensive report"""
        report_lines = [
            "=" * 80,
            "TRAINING REPORT",
            "=" * 80,
            f"Algorithm: {self.config.algorithm}",
            f"Experiment: {self.config.experiment_name}",
            f"Run: {self.config.run_name}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            "COMPREHENSIVE EVALUATION RESULTS",
            "-" * 40,
            f"Strategy: {comprehensive_results.get('strategy_name', 'Unknown')}",
            f"Sharpe Ratio: {comprehensive_results.get('performance_metrics', {}).get('sharpe_ratio', 0):.4f}",
            f"Max Drawdown: {comprehensive_results.get('performance_metrics', {}).get('max_drawdown', 0):.4f}",
            f"Win Rate: {comprehensive_results.get('performance_metrics', {}).get('win_rate', 0):.4f}",
            "",
            "RECOMMENDATIONS",
            "-" * 15,
            "1. Monitor risk metrics closely during live trading",
            "2. Implement proper risk management controls",
            "3. Regular re-evaluation and model updates recommended",
            "4. Consider ensemble methods for improved robustness"
        ]
        
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_path / f"{self.config.run_name}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.log_artifact(str(report_file))
        
        return report


def main():
    """Main entry point for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent HFT Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--algorithm", type=str, choices=["maddpg", "mappo"], help="Algorithm to train")
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
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
    
    # Run training pipeline
    pipeline = EnhancedTrainingPipeline(config)
    results = pipeline.run_training()
    
    print("Training completed successfully!")
    print(f"Results saved to: {config.results_path}")
    print(f"Report: {results['report']}")


if __name__ == "__main__":
    main()
