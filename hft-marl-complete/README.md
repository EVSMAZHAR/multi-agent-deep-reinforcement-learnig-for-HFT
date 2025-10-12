# Multi-Agent Deep Reinforcement Learning for High-Frequency Trading

A comprehensive implementation of multi-agent reinforcement learning algorithms for high-frequency trading, based on the thesis "Multi-agent deep reinforcement learning for high frequency trading".

## 🚀 Features

### Data Pipeline ⭐ NEW
- **Data Collection**: Automated ingestion from simulator outputs or custom sources
- **Feature Engineering**: Market microstructure features with technical indicators
- **Temporal Sequences**: Automatic creation of time-series data for training
- **Scaling & Normalization**: Robust scaling for financial data
- **Synthetic Data**: Automatic fallback generation for testing
- **Configuration**: Flexible YAML-based configuration system

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
│   ├── data/                       # Data collection modules
│   │   ├── ingest.py               # Raw data ingestion
│   │   └── make_dataset.py         # Dataset creation
│   ├── features/                   # Feature engineering
│   │   └── build_features.py       # Feature computation
│   ├── marl/
│   │   ├── env_enhanced.py         # Enhanced CTDE environment
│   │   └── policies/
│   │       ├── enhanced_maddpg.py  # Enhanced MADDPG implementation
│   │       └── enhanced_mappo.py   # Enhanced MAPPO implementation
│   ├── baselines/
│   │   └── comprehensive_baselines.py  # Baseline strategies
│   ├── evaluation/
│   │   └── comprehensive_evaluation.py # Evaluation framework
│   └── training/
│       └── enhanced_training.py    # Training pipeline
├── configs/
│   ├── data.yaml                   # Data collection config
│   ├── features.yaml               # Feature engineering config
│   ├── training_config.yaml        # Training configuration
│   └── environment_config.yaml     # Environment configuration
├── scripts/
│   └── prepare_data.sh             # Data pipeline script
├── data/                           # Data directory
│   ├── sim/                        # Raw simulator data
│   ├── interim/                    # Processed snapshots
│   └── features/                   # Engineered features
├── models/                         # Trained models
├── results/                        # Experiment results
├── logs/                           # Training logs
├── tests/                          # Test suite
│   └── test_data_pipeline.py       # Data pipeline tests
├── main.py                         # Main entry point
├── requirements.txt                # Dependencies
├── DATA_PIPELINE.md                # Data pipeline docs
├── INTEGRATION_SUMMARY.md          # Integration details
└── README.md                       # This file
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

### 1. Prepare Data (Optional)

The training pipeline automatically prepares data, but you can do it manually:

```bash
# Automatic pipeline execution (recommended)
./scripts/prepare_data.sh

# Or run individual steps
python src/data/ingest.py --config configs/data.yaml
python src/features/build_features.py --config configs/features.yaml
python src/data/make_dataset.py --config configs/data.yaml
```

**Note**: If no raw data exists, synthetic market data will be generated automatically.

### 2. Train a Model

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

### 3. Evaluate a Model

```bash
python main.py evaluate --model models/maddpg_final.pt --episodes 100
```

### 4. Run Baseline Strategies

```bash
# Avellaneda-Stoikov market making
python main.py baseline --strategy avellaneda-stoikov --episodes 100

# TWAP execution
python main.py baseline --strategy twap --episodes 100

# Momentum strategy
python main.py baseline --strategy momentum --episodes 100
```

### 5. Compare Models

```bash
python main.py compare --models models/maddpg_final.pt models/mappo_final.pt --episodes 100
```

## 📊 Configuration

### Data Collection Configuration

Edit `configs/data.yaml` to customize data pipeline:

```yaml
# Timezone and symbols
timezone: UTC
symbols: [SYMA, SYMB]

# Temporal settings
decision_ms: 100
history_T: 20

# Data paths
paths:
  sim: data/sim
  interim: data/interim
  features: data/features

# Train/validation/test splits
splits:
  dev:
    start: '2020-01-01'
    end: '2021-12-31'
  val:
    start: '2022-01-01'
    end: '2022-06-30'
  test:
    start: '2022-07-01'
    end: '2022-12-31'

# Scaling method
scaling:
  method: robust  # or 'standard'
```

### Feature Engineering Configuration

Edit `configs/features.yaml` to customize features:

```yaml
# Feature computation
history_T: 20
decision_ms: 100

# Technical indicator windows
windows:
  fast: 10
  slow: 30
  ofi_ms: 1000
  realised_vol_ms: 2000

# Features to compute
aux:
  - spread
  - microprice
  - imbalance
  - returns
  - volatility_fast
  - volatility_slow

# Scaler configuration
scaler:
  type: robust  # 'robust' or 'standard'
  fit_on: dev
```

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

### Custom Data Pipeline

```python
from src.training.enhanced_training import DataManager, TrainingConfig

# Create configuration
config = TrainingConfig(
    data_path="data",
    features_path="data/features"
)

# Initialize data manager
data_manager = DataManager(config)

# Prepare data (with optional force rebuild)
data_manager.prepare_data(force_rebuild=False)

# Or run individual pipeline steps
data_manager._create_synthetic_market_data()  # Generate synthetic data
data_manager._build_features()                # Build features
data_manager._create_datasets()              # Create train/val/test splits
```

### Adding Custom Features

```python
# Edit src/features/build_features.py
def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add your custom feature computation
    df['my_feature'] = ...  # Your calculation
    return df

# Update configs/features.yaml
aux:
  - my_feature
```

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

Run the complete test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
# Test data pipeline
python tests/test_data_pipeline.py

# Test environment
pytest tests/test_sanity.py -v

# Test specific components
pytest tests/test_environment.py -v
pytest tests/test_algorithms.py -v
pytest tests/test_baselines.py -v
```

## 📚 Documentation

### Comprehensive Guides

- **[DATA_PIPELINE.md](DATA_PIPELINE.md)**: Complete data pipeline documentation
- **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)**: Integration details and migration guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Overall implementation summary

### API Reference

- **Data Pipeline**: `src/data/` and `src/features/`
- **Environment**: `src/marl/env_enhanced.py`
- **Algorithms**: `src/marl/policies/`
- **Baselines**: `src/baselines/comprehensive_baselines.py`
- **Evaluation**: `src/evaluation/comprehensive_evaluation.py`
- **Training**: `src/training/enhanced_training.py`

### Examples

See the `examples/` directory for detailed usage examples:
- Basic training example
- Custom data pipeline setup
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
