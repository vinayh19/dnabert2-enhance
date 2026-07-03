# Reproducing DNABERT-2 Enhance

## Environment
- Python 3.10
- CUDA 12.8
- Hardware tested: NVIDIA B200 × [N]
- Install:
    - python -m venv .venv && source .venv/bin/activate
    - pip install -r requirements.txt

- flash-attn often needs no-build-isolation:
  - pip install flash-attn==<version> --no-build-isolation

## Data
Download the DNABERT-2 pretraining corpus (multi-species, 135 genomes) from
the release linked in the DNABERT-2 repository:
https://github.com/MAGICS-LAB/DNABERT_2

Convert to MosaicML streaming shards format. [Reference the shard-conversion
script or point to the streaming library docs:
https://github.com/mosaicml/streaming]

Set `data_local` in the yaml to the shards directory.

## Pretraining commands

Three model variants:

- 170-token context, 30% train / 15% eval mask
  
composer -n <NGPUS> src/main.py configs/pretrain_ctx170_mlm30-15.yaml

- 256-token context, 15% train / 15% eval mask
  
composer -n <NGPUS> src/main.py configs/pretrain_ctx256_mlm15-15.yaml

- 256-token context, 30% train / 30% eval mask
  
composer -n <NGPUS> src/main.py configs/pretrain_ctx256_mlm30-30.yaml

## Compute budget (as run)
- Hardware: NVIDIA B200 × [N]
- Wall-clock per run: ~28 hours
- Batches: 250,000
- Approximate epochs over corpus: ~70
- Global batch size: 8192
- Peak LR: 7.07e-4
- Precision: amp_bf16
- Seed: 42

## Expected outputs
- Checkpoints in `save_folder`, one every `save_interval`
- Best-eval-loss checkpoint at `<save_folder>/best/best-loss-checkpoint.pt`
- WandB run in project `<your-project>`
