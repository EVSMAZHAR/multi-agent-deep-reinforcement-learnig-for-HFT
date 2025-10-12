# HFT-MARL: Phase 4–7

Adds:
- **MADDPG** (replay, targets, TD3-style noise/delays)
- **Baselines**: Avellaneda–Stoikov market maker; single-agent PPO stub preserved
- **Env**: More realistic fills & transient price impact
- **Evaluation**: Multi-seed CI, risk metrics; plots & tables

## Pipelines
```bash
# Data → features → dataset (same as before)
python -m src.sim.run_abides --config configs/sim.yaml --out data/sim/
python -m src.sim.run_jaxlob --config configs/sim.yaml --out data/sim/
python -m src.data.ingest --config configs/data.yaml
python -m src.features.build_features --config configs/features.yaml
python -m src.data.make_dataset --config configs/data.yaml

# Train MAPPO (working PPO)
python -m src.training.train_mappo --config configs/mappo.yaml

# Train MADDPG (new)
python -m src.training.train_maddpg --config configs/maddpg.yaml

# Evaluate with baselines + CI
python -m src.training.eval --config configs/eval.yaml    # RL models
python -m src.training.eval_baselines --config configs/eval.yaml  # baselines

# Make figures & tables
python -m src.reports.make_figures
python -m src.reports.make_tables
```
