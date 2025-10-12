# HFT-MARL: Risk-aware Multi-Agent RL for High-Frequency Trading (Phase 1–3)

This package adds **richer features (OFI, realised vol, queue proxies)**, a **CTDE multi-agent env** driven by dataset tensors,
and a **working MAPPO training loop** with GAE + PPO clipping.

## Pipeline
```bash
# 0) (optional) synthetic data from simulators
python -m src.sim.run_abides --config configs/sim.yaml --out data/sim/
python -m src.sim.run_jaxlob --config configs/sim.yaml --out data/sim/

# 1) Ingest & clean
python -m src.data.ingest --config configs/data.yaml

# 2) Build features (enhanced)
python -m src.features.build_features --config configs/features.yaml

# 3) Make dataset (rolling windows + scalers)
python -m src.data.make_dataset --config configs/data.yaml

# 4) Train MAPPO (working loop)
python -m src.training.train_mappo --config configs/mappo.yaml

# 5) Evaluate
python -m src.training.eval --config configs/eval.yaml
```

> Note: The simulation and env are **research-grade simplifications** so you can learn/iterate. We will refine fills/impact and add ABIDES/JAX-LOB connectors later.
