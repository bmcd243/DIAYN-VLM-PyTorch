# DIAYN-VLM Configuration System

## Quick Start

### 1. Run with a config file:
```bash
python main.py --config configs/halfcheetah_vit_b32.yaml
```

### 2. Override specific parameters:
```bash
python main.py --config configs/halfcheetah_vit_l14.yaml --n_skills 50 --batch_size 1024
```

### 3. Use command-line only (backward compatible):
```bash
python main.py --env_name="HalfCheetah-v5" --do_train --train_from_scratch --n_skills=20
```

## Available Config Files

| Config File | CLIP Model | Speed | Use Case |
|------------|------------|-------|----------|
| `configs/base.yaml` | ViT-B/32 | Fast | Default settings |
| `configs/halfcheetah_vit_b32.yaml` | ViT-B/32 | Fast | Quick experiments |
| `configs/halfcheetah_vit_l14.yaml` | ViT-L/14 | Slow | Best quality |
| `configs/humanoid.yaml` | ViT-B/32 | Fast | Complex environment |

## Example Commands

### Fast experimentation (3000 episodes, ViT-B/32):
```bash
python main.py --config configs/halfcheetah_vit_b32.yaml
```

### High-quality training (5000 episodes, ViT-L/14):
```bash
python main.py --config configs/halfcheetah_vit_l14.yaml
```

### Quick ablation (override n_skills):
```bash
python main.py --config configs/base.yaml --n_skills 10
python main.py --config configs/base.yaml --n_skills 20
python main.py --config configs/base.yaml --n_skills 50
```

### Enable logging:
```bash
python main.py --config configs/halfcheetah_vit_b32.yaml --use_wandb
```

### Train on multiple GPUs with different seeds:
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/halfcheetah_vit_b32.yaml --seed 123 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/halfcheetah_vit_b32.yaml --seed 456 &

# GPU 2
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/halfcheetah_vit_b32.yaml --seed 789 &
```

## Benefits of This System

✅ **Easy to read**: YAML files are human-readable and well-documented  
✅ **Version control friendly**: Track experiment configs in git  
✅ **Reproducible**: Share exact configs with collaborators  
✅ **Flexible**: Override any parameter via command-line  
✅ **Backward compatible**: Still works with pure CLI arguments  
✅ **Organized**: All configs in one place

## Creating Custom Configs

1. Copy an existing config (e.g., `configs/base.yaml`)
2. Modify the parameters you need
3. Save with a descriptive name (e.g., `my_experiment.yaml`)
4. Run: `python main.py --config configs/my_experiment.yaml`

Example custom config:
```yaml
# configs/my_experiment.yaml
env_name: "HalfCheetah-v5"
n_skills: 30
batch_size: 1024
clip_model: "ViT-L/14"
embedding_dim: 768
use_wandb: true
```
