#!/usr/bin/env bash
set -e
python -m src.sim.run_abides --config configs/sim.yaml --out data/sim/
python -m src.sim.run_jaxlob --config configs/sim.yaml --out data/sim/
python -m src.data.ingest --config configs/data.yaml
python -m src.features.build_features --config configs/features.yaml
python -m src.data.make_dataset --config configs/data.yaml
python -m src.training.train_mappo --config configs/mappo.yaml
python -m src.training.eval --config configs/eval.yaml
