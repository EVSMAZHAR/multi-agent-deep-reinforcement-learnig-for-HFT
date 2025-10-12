# Multi-Agent Deep Reinforcement Learning for High-Frequency Trading

A comprehensive implementation of multi-agent reinforcement learning algorithms for high-frequency trading, based on the thesis "Multi-agent deep reinforcement learning for high frequency trading".

## 🚀 Features

### Core Algorithms
- **Enhanced MADDPG**: Multi-Agent Deep Deterministic Policy Gradient with advanced features
- **Enhanced MAPPO**: Multi-Agent Proximal Policy Optimization with coordination mechanisms
- **Centralized Training, Decentralized Execution (CTDE)** framework

### Market Simulation
- **Realistic Market Microstructure**: Advanced order book simulation with market impact
- **Risk Management**: Comprehensive risk controls and limits
- **Multi-Agent Coordination**: Sophisticated agent interaction mechanisms

### Baseline Strategies
- **Avellaneda-Stoikov Market Making**: Optimal market making strategy
- **TWAP/VWAP/POV**: Execution algorithms
- **Momentum/Mean Reversion**: Technical trading strategies
- **Technical Analysis**: RSI, SMA, and other indicators

### Evaluation Framework
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios, VaR, CVaR
- **Statistical Testing**: Bootstrap confidence intervals, significance tests
- **Multi-Agent Metrics**: Coordination efficiency, diversity indices
- **Risk Analysis**: Drawdown analysis, tail risk metrics

## 📁 Project Structure

```
hft-marl-complete/
├── src/
│   ├── marl/
│   │   ├── env_enhanced.py          # Enhanced CTDE environment
│   │   └── policies/
│   │       ├── enhanced_maddpg.py   # Enhanced MADDPG implementation
│   │       └── enhanced_mappo.py    # Enhanced MAPPO implementation
│   ├── baselines/
│   │   └── comprehensive_baselines.py  # Baseline strategies
│   ├── evaluation/
│   │   └── comprehensive_evaluation.py # Evaluation framework
│   └── training/
│       └── enhanced_training.py     # Training pipeline
├── configs/
│   ├── training_config.yaml        # Training configuration
│   └── environment_config.yaml     # Environment configuration
├── data/                           # Data directory
├── models/                         # Trained models
├── results/                        # Experiment results
├── logs/                          # Training logs
├── main.py                        # Main entry point
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hft-marl-complete
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Setup directories**:
```bash
python main.py train --episodes 1  # This will create necessary directories
```

## 🚀 Quick Start

### 1. Train a Model

**Train MADDPG**:
```bash
python main.py train --algorithm maddpg --episodes 10000
```

**Train MAPPO**:
```bash
python main.py train --algorithm mappo --episodes 10000
```

**Use configuration file**:
```bash
python main.py train --config configs/training_config.yaml
```

### 2. Evaluate a Model

```bash
python main.py evaluate --model models/maddpg_final.pt --episodes 100
```

### 3. Run Baseline Strategies

```bash
# Avellaneda-Stoikov market making
python main.py baseline --strategy avellaneda-stoikov --episodes 100

# TWAP execution
python main.py baseline --strategy twap --episodes 100

# Momentum strategy
python main.py baseline --strategy momentum --episodes 100
```

### 4. Compare Models

```bash
python main.py compare --models models/maddpg_final.pt models/mappo_final.pt --episodes 100
```

## 📊 Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize:

```yaml
# Algorithm selection
algorithm: "maddpg"  # or "mappo"

# Training parameters
total_episodes: 10000
eval_frequency: 100
save_frequency: 500

# MADDPG specific
maddpg_config:
  lr_actor: 0.0001
  lr_critic: 0.001
  gamma: 0.99
  tau: 0.005
  batch_size: 256
  buffer_size: 1000000
  noise_scale: 0.1
  noise_decay: 0.9995

# MAPPO specific
mappo_config:
  lr_actor: 0.0003
  lr_critic: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.01
  epochs: 4
  batch_size: 64
```

### Environment Configuration

Edit `configs/environment_config.yaml` to customize:

```yaml
# Market parameters
market_config:
  tick_size: 0.01
  decision_ms: 100
  volatility: 0.02
  mean_reversion_speed: 0.05

# Risk management
risk_config:
  max_inventory: 1000
  max_drawdown: 10000.0
  max_position_value: 50000.0

# Reward function
reward_config:
  lambda_inventory: 0.5
  lambda_impact: 0.2
  lambda_spread: 0.1
  lambda_risk: 1.0
```

## 🔬 Advanced Usage

### Custom Environment

```python
from src.marl.env_enhanced import EnhancedCTDEHFTEnv, MarketConfig, RiskConfig, RewardConfig

# Create custom configurations
market_config = MarketConfig(
    tick_size=0.01,
    volatility=0.02,
    mean_reversion_speed=0.05
)

risk_config = RiskConfig(
    max_inventory=1000,
    max_drawdown=10000.0
)

reward_config = RewardConfig(
    lambda_inventory=0.5,
    lambda_impact=0.2
)

# Create environment
env = EnhancedCTDEHFTEnv(
    dataset_path="data/features/dev_tensors.npz",
    scaler_path="data/features/scaler.json",
    market_config=market_config,
    risk_config=risk_config,
    reward_config=reward_config,
    episode_len=1000
)
```

### Custom Algorithm

```python
from src.marl.policies.enhanced_maddpg import EnhancedMADDPG, MADDPGConfig

# Create custom configuration
config = MADDPGConfig(
    lr_actor=1e-4,
    lr_critic=1e-3,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    buffer_size=1000000,
    use_prioritized_replay=True,
    use_double_q=True
)

# Create algorithm
maddpg = EnhancedMADDPG(
    obs_dims=obs_dims,
    action_dims=action_dims,
    agent_ids=agent_ids,
    config=config,
    device='cuda'
)
```

### Custom Baseline Strategy

```python
from src.baselines.comprehensive_baselines import BaseStrategy, StrategyType

class CustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Custom Strategy", StrategyType.MARKET_MAKING)
    
    def generate_orders(self, market_state, agent_state):
        # Implement your custom strategy logic
        orders = []
        # ... strategy implementation
        return orders
    
    def update_state(self, execution_result):
        # Update strategy state
        pass
```

## 📈 Results and Evaluation

### Performance Metrics

The system provides comprehensive performance evaluation:

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown
- **Multi-Agent Metrics**: Coordination efficiency, diversity indices
- **Statistical Significance**: Bootstrap confidence intervals, t-tests

### Experiment Tracking

Results are automatically logged to:
- **MLflow**: Experiment tracking and model registry
- **Logs**: Detailed training logs in `logs/` directory
- **Results**: Comprehensive results in `results/` directory

### Visualization

Generate performance comparison plots:

```python
from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
evaluator.plot_performance_comparison(
    strategy_results=results,
    save_path="results/performance_comparison.png"
)
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
pytest tests/test_environment.py -v
pytest tests/test_algorithms.py -v
pytest tests/test_baselines.py -v
```

## 📚 Documentation

### API Reference

- **Environment**: `src/marl/env_enhanced.py`
- **Algorithms**: `src/marl/policies/`
- **Baselines**: `src/baselines/comprehensive_baselines.py`
- **Evaluation**: `src/evaluation/comprehensive_evaluation.py`

### Examples

See the `examples/` directory for detailed usage examples:
- Basic training example
- Custom environment setup
- Baseline strategy implementation
- Evaluation and comparison

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{hft_marl_thesis,
  title={Multi-agent deep reinforcement learning for high frequency trading},
  author={Your Name},
  year={2024},
  institution={Your University},
  type={Master's Thesis}
}
```

## 🙏 Acknowledgments

- ABIDES framework for market simulation
- OpenAI Gym for reinforcement learning environment
- PyTorch for deep learning framework
- MLflow for experiment tracking

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples

---

**Note**: This is a research implementation. Use at your own risk in production environments. Always implement proper risk management and testing before deploying to live trading systems.
