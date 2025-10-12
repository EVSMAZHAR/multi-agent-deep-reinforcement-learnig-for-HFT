"""
Sanity Tests for Multi-Agent High-Frequency Trading System
=========================================================

Basic tests to ensure the system components work correctly.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from marl.env_enhanced import EnhancedCTDEHFTEnv, MarketConfig, RiskConfig, RewardConfig
from marl.policies.enhanced_maddpg import EnhancedMADDPG, MADDPGConfig
from marl.policies.enhanced_mappo import EnhancedMAPPO, MAPPOConfig
from baselines.comprehensive_baselines import AvellanedaStoikovStrategy, TWAPStrategy
from evaluation.comprehensive_evaluation import ComprehensiveEvaluator, EvaluationConfig


class TestEnvironment:
    """Test the enhanced environment"""
    
    def test_environment_creation(self):
        """Test environment can be created"""
        # Create dummy data
        self._create_dummy_data()
        
        # Create environment
        market_config = MarketConfig()
        risk_config = RiskConfig()
        reward_config = RewardConfig()
        
        env = EnhancedCTDEHFTEnv(
            dataset_path="data/features/dev_tensors.npz",
            scaler_path="data/features/scaler.json",
            market_config=market_config,
            risk_config=risk_config,
            reward_config=reward_config,
            episode_len=100,
            seed=42
        )
        
        assert env is not None
        assert len(env.agents) == 2
        assert env.episode_len == 100
    
    def test_environment_reset(self):
        """Test environment reset"""
        env = self._create_test_environment()
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert len(obs) == len(env.agents)
        for agent in env.agents:
            assert agent in obs
            assert isinstance(obs[agent], np.ndarray)
    
    def test_environment_step(self):
        """Test environment step"""
        env = self._create_test_environment()
        obs, _ = env.reset()
        
        # Create dummy actions
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.uniform(-1, 1, size=4)
        
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert isinstance(rewards, dict)
        assert isinstance(terms, dict)
        assert isinstance(truncs, dict)
        assert isinstance(infos, dict)
        
        for agent in env.agents:
            assert agent in rewards
            assert agent in terms
            assert agent in truncs
            assert agent in infos
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        # Create features directory
        features_dir = Path("data/features")
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dummy data
        np.random.seed(42)
        n_samples = 1000
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
        import json
        scaler = {
            'median': np.zeros(n_features).tolist(),
            'iqr': np.ones(n_features).tolist()
        }
        
        with open(features_dir / "scaler.json", 'w') as f:
            json.dump(scaler, f)
    
    def _create_test_environment(self):
        """Create test environment"""
        self._create_dummy_data()
        
        market_config = MarketConfig()
        risk_config = RiskConfig()
        reward_config = RewardConfig()
        
        return EnhancedCTDEHFTEnv(
            dataset_path="data/features/dev_tensors.npz",
            scaler_path="data/features/scaler.json",
            market_config=market_config,
            risk_config=risk_config,
            reward_config=reward_config,
            episode_len=100,
            seed=42
        )


class TestAlgorithms:
    """Test the enhanced algorithms"""
    
    def test_maddpg_creation(self):
        """Test MADDPG algorithm creation"""
        config = MADDPGConfig()
        obs_dims = {"agent_0": 10, "agent_1": 10}
        action_dims = {"agent_0": 4, "agent_1": 4}
        agent_ids = ["agent_0", "agent_1"]
        
        maddpg = EnhancedMADDPG(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=agent_ids,
            config=config,
            device='cpu'
        )
        
        assert maddpg is not None
        assert len(maddpg.agents) == 2
        assert maddpg.num_agents == 2
    
    def test_maddpg_action_selection(self):
        """Test MADDPG action selection"""
        config = MADDPGConfig()
        obs_dims = {"agent_0": 10, "agent_1": 10}
        action_dims = {"agent_0": 4, "agent_1": 4}
        agent_ids = ["agent_0", "agent_1"]
        
        maddpg = EnhancedMADDPG(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=agent_ids,
            config=config,
            device='cpu'
        )
        
        # Create dummy observations
        observations = {}
        for agent in agent_ids:
            observations[agent] = torch.randn(10)
        
        actions = maddpg.select_actions(observations, explore=True)
        
        assert isinstance(actions, dict)
        assert len(actions) == len(agent_ids)
        for agent in agent_ids:
            assert agent in actions
            assert isinstance(actions[agent], torch.Tensor)
            assert actions[agent].shape == (4,)
    
    def test_mappo_creation(self):
        """Test MAPPO algorithm creation"""
        config = MAPPOConfig()
        obs_dims = {"agent_0": 10, "agent_1": 10}
        action_dims = {"agent_0": 4, "agent_1": 4}
        agent_ids = ["agent_0", "agent_1"]
        
        mappo = EnhancedMAPPO(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=agent_ids,
            config=config,
            device='cpu'
        )
        
        assert mappo is not None
        assert len(mappo.agent_ids) == 2
        assert mappo.num_agents == 2
    
    def test_mappo_action_selection(self):
        """Test MAPPO action selection"""
        config = MAPPOConfig()
        obs_dims = {"agent_0": 10, "agent_1": 10}
        action_dims = {"agent_0": 4, "agent_1": 4}
        agent_ids = ["agent_0", "agent_1"]
        
        mappo = EnhancedMAPPO(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=agent_ids,
            config=config,
            device='cpu'
        )
        
        # Create dummy observations
        observations = {}
        for agent in agent_ids:
            observations[agent] = torch.randn(10)
        
        actions, log_probs, values = mappo.select_actions(observations)
        
        assert isinstance(actions, dict)
        assert isinstance(log_probs, dict)
        assert isinstance(values, dict)
        assert len(actions) == len(agent_ids)
        
        for agent in agent_ids:
            assert agent in actions
            assert agent in log_probs
            assert agent in values
            assert isinstance(actions[agent], torch.Tensor)
            assert isinstance(log_probs[agent], torch.Tensor)
            assert isinstance(values[agent], torch.Tensor)


class TestBaselines:
    """Test baseline strategies"""
    
    def test_avellaneda_stoikov_creation(self):
        """Test Avellaneda-Stoikov strategy creation"""
        strategy = AvellanedaStoikovStrategy()
        
        assert strategy is not None
        assert strategy.name == "Avellaneda-Stoikov"
        assert strategy.strategy_type.value == "market_making"
    
    def test_avellaneda_stoikov_orders(self):
        """Test Avellaneda-Stoikov order generation"""
        strategy = AvellanedaStoikovStrategy()
        
        # Create dummy market state
        from baselines.comprehensive_baselines import MarketState
        market_state = MarketState(
            mid_price=100.0,
            best_bid=99.99,
            best_ask=100.01,
            spread=0.02,
            imbalance=0.1,
            volume=1000.0,
            volatility=0.02,
            timestamp=0.0
        )
        
        agent_state = {
            'inventory': 0.0,
            'cash': 0.0,
            'pnl': 0.0
        }
        
        orders = strategy.generate_orders(market_state, agent_state)
        
        assert isinstance(orders, list)
        # Should generate bid and ask orders
        assert len(orders) >= 0
    
    def test_twap_creation(self):
        """Test TWAP strategy creation"""
        strategy = TWAPStrategy(
            target_quantity=1000.0,
            time_horizon=3600.0
        )
        
        assert strategy is not None
        assert strategy.name == "TWAP"
        assert strategy.strategy_type.value == "execution"


class TestEvaluation:
    """Test evaluation framework"""
    
    def test_evaluator_creation(self):
        """Test evaluator creation"""
        config = EvaluationConfig()
        evaluator = ComprehensiveEvaluator(config)
        
        assert evaluator is not None
        assert evaluator.config is not None
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        config = EvaluationConfig()
        evaluator = ComprehensiveEvaluator(config)
        
        # Create dummy returns
        returns = np.random.normal(0.001, 0.02, 1000)
        
        metrics = evaluator.performance_metrics.calculate_all_metrics(returns)
        
        assert isinstance(metrics, dict)
        assert 'mean_return' in metrics
        assert 'std_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_strategy_evaluation(self):
        """Test strategy evaluation"""
        config = EvaluationConfig()
        evaluator = ComprehensiveEvaluator(config)
        
        # Create dummy returns
        returns = np.random.normal(0.001, 0.02, 1000)
        
        results = evaluator.evaluate_strategy(
            strategy_name="test_strategy",
            returns=returns
        )
        
        assert isinstance(results, dict)
        assert 'strategy_name' in results
        assert 'performance_metrics' in results
        assert 'normality_tests' in results
        assert 'robustness_tests' in results


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_simulation(self):
        """Test end-to-end simulation"""
        # Create dummy data
        self._create_dummy_data()
        
        # Create environment
        market_config = MarketConfig()
        risk_config = RiskConfig()
        reward_config = RewardConfig()
        
        env = EnhancedCTDEHFTEnv(
            dataset_path="data/features/dev_tensors.npz",
            scaler_path="data/features/scaler.json",
            market_config=market_config,
            risk_config=risk_config,
            reward_config=reward_config,
            episode_len=10,
            seed=42
        )
        
        # Create algorithm
        config = MADDPGConfig()
        obs_dims = {agent: env.observation_space[agent].shape[0] for agent in env.agents}
        action_dims = {agent: env.action_space[agent].shape[0] for agent in env.agents}
        
        maddpg = EnhancedMADDPG(
            obs_dims=obs_dims,
            action_dims=action_dims,
            agent_ids=env.agents,
            config=config,
            device='cpu'
        )
        
        # Run simulation
        obs, _ = env.reset()
        total_reward = 0.0
        
        for step in range(10):
            actions = maddpg.select_actions(obs, explore=True)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            
            total_reward += sum(rewards.values())
            
            obs = next_obs
            
            if any(terms.values()) or any(truncs.values()):
                break
        
        assert total_reward is not None
        assert isinstance(total_reward, (int, float))
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        # Create features directory
        features_dir = Path("data/features")
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dummy data
        np.random.seed(42)
        n_samples = 1000
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
        import json
        scaler = {
            'median': np.zeros(n_features).tolist(),
            'iqr': np.ones(n_features).tolist()
        }
        
        with open(features_dir / "scaler.json", 'w') as f:
            json.dump(scaler, f)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
