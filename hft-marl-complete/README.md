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
│   ├── sim/                        # Market simulation
│   │   ├── run_abides.py          # ABIDES-like simulator
│   │   └── run_jaxlob.py          # JAX-LOB-like simulator
│   ├── data/                       # Data processing
│   │   ├── ingest.py              # Data ingestion
│   │   └── make_dataset.py        # Dataset preparation
│   ├── features/                   # Feature engineering
│   │   └── build_features.py      # Feature builder
│   ├── marl/
│   │   ├── env_enhanced.py        # Enhanced CTDE environment
│   │   └── policies/
│   │       ├── enhanced_maddpg.py # Enhanced MADDPG implementation
│   │       └── enhanced_mappo.py  # Enhanced MAPPO implementation
│   ├── baselines/
│   │   └── comprehensive_baselines.py  # Baseline strategies
│   ├── evaluation/
│   │   └── comprehensive_evaluation.py # Evaluation framework
│   └── training/
│       └── enhanced_training.py   # Training pipeline
├── configs/
│   ├── data_pipeline.yaml         # Data pipeline configuration
│   ├── features.yaml              # Feature engineering configuration
│   ├── training_config.yaml       # Training configuration
│   └── environment_config.yaml    # Environment configuration
├── data/                          # Data directory
│   ├── sim/                       # Raw simulator output
│   ├── interim/                   # Intermediate data
│   └── features/                  # Engineered features & tensors
├── models/                        # Trained models
├── results/                       # Experiment results
├── logs/                         # Training logs
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
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

4. **Prepare training data**:
```bash
python main.py prepare-data  # This will generate synthetic market data and features
```

## 🚀 Quick Start

### 1. Prepare Training Data

**First, generate and prepare the training data**:
```bash
python main.py prepare-data --config configs/data_pipeline.yaml
```

This command will:
- Run market simulators (ABIDES and JAX-LOB) to generate synthetic order book data
- Ingest and consolidate market snapshots
- Engineer trading features (spread, imbalance, microprice, etc.)
- Create training-ready time-series tensors

The data preparation pipeline generates:
- `data/sim/*_snapshots*.parquet` - Raw market snapshots
- `data/interim/snapshots.parquet` - Consolidated snapshots
- `data/features/features.parquet` - Engineered features
- `data/features/scaler.json` - Feature scaling parameters
- `data/features/*_tensors.npz` - Training-ready tensors (format: [N, T, F])

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

## 📊 Data Pipeline

The data pipeline consists of four stages:

### Stage 1: Market Simulation
Generate synthetic order book snapshots using two simulators:
- **ABIDES Simulator**: Ornstein-Uhlenbeck price process with realistic market dynamics
- **JAX-LOB Simulator**: Jump-diffusion price process with tight spreads

Configuration in `configs/data_pipeline.yaml`:
```yaml
symbols: ['SYMA']
num_samples: 10000
tick_ms: 100
volatility: 0.02
spread_target_bps: 5
```

### Stage 2: Data Ingestion
Consolidate raw market snapshots from multiple simulators:
- Merge snapshots from different sources
- Sort by symbol and timestamp
- Remove duplicates
- Output: `data/interim/snapshots.parquet`

### Stage 3: Feature Engineering
Extract trading features from raw market data:
- **Basic features**: spread, imbalance, microprice
- **Technical indicators**: returns, volatility, price components
- **Robust scaling**: Median and IQR-based normalization
- Output: `data/features/features.parquet`, `data/features/scaler.json`

Engineered features (12 total):
- `best_bid`, `best_ask` - Top of book prices
- `bid_qty_1`, `ask_qty_1` - Top of book quantities
- `spread` - Best ask - best bid
- `imbalance` - Order book imbalance
- `microprice` - Volume-weighted mid-price
- `mid_price` - Simple mid-price
- `returns` - Price returns
- `volatility` - Rolling volatility
- `bid_value`, `ask_value` - Price × quantity

### Stage 4: Dataset Preparation
Create time-series tensors for training:
- **Sliding window**: History of T=20 timesteps
- **Format**: [N, T, F] where N=samples, T=timesteps, F=features
- **Data splits**: train/dev/val/test by date ranges
- Output: `data/features/*_tensors.npz`

### Running Individual Pipeline Steps

You can also run each step individually:

```bash
# Step 1: Generate market data
python -m src.sim.run_abides --config configs/data_pipeline.yaml --out data/sim
python -m src.sim.run_jaxlob --config configs/data_pipeline.yaml --out data/sim

# Step 2: Ingest data
python -m src.data.ingest --config configs/data_pipeline.yaml

# Step 3: Build features
python -m src.features.build_features --config configs/data_pipeline.yaml

# Step 4: Prepare datasets
python -m src.data.make_dataset --config configs/data_pipeline.yaml
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
