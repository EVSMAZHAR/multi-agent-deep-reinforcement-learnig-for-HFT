# HFT-MARL: Risk-aware Multi-Agent RL for High-Frequency Trading (Phase 0)

End-to-end scaffold for a CTDE-based, risk-aware MARL system for HFT.

## Quick start
```bash
conda create -n hft-marl python=3.10 -y
conda activate hft-marl
pip install -r requirements.txt

python -m src.sim.run_abides --config configs/sim.yaml --out data/sim/
python -m src.sim.run_jaxlob --config configs/sim.yaml --out data/sim/
python -m src.data.ingest --config configs/data.yaml
python -m src.features.build_features --config configs/features.yaml
python -m src.data.make_dataset --config configs/data.yaml

python -m src.training.train_mappo --config configs/mappo.yaml
python -m src.training.eval --config configs/eval.yaml
```
