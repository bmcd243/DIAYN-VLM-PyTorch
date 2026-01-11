# Configuration Files

This directory contains YAML configuration files for different DIAYN-VLM experiments.

## File Structure

- **`base.yaml`**: Default configuration with all parameters documented
- **`halfcheetah_vit_b32.yaml`**: Fast training config with ViT-B/32 (optimized for speed)
- **`halfcheetah_vit_l14.yaml`**: High-quality config with ViT-L/14 (better embeddings)
- **`humanoid.yaml`**: Configuration for complex Humanoid environment

## Usage

### 1. Use a config file directly:
```bash
python main.py --config configs/halfcheetah_vit_l14.yaml
```

### 2. Use a config file and override specific parameters:
```bash
python main.py --config configs/halfcheetah_vit_b32.yaml --n_skills 50 --batch_size 1024
```

### 3. Use only command-line arguments (backward compatible):
```bash
python main.py --env_name="HalfCheetah-v5" --do_train --train_from_scratch --n_skills=20
```

### 4. Enable Weights & Biases logging:
```bash
python main.py --config configs/halfcheetah_vit_l14.yaml --use_wandb
```

## Priority Order

Configuration values are loaded in this order (later values override earlier ones):

1. **Base defaults** (`base.yaml`)
2. **Specific config file** (via `--config`)
3. **Command-line arguments** (highest priority)

## Creating New Configs

To create a new experiment config:

1. Copy `base.yaml` to a new file (e.g., `my_experiment.yaml`)
2. Modify only the parameters you want to change
3. Add comments to document your choices
4. Run with: `python main.py --config configs/my_experiment.yaml`

## CLIP Model Options

Available CLIP models and their embedding dimensions:

| Model | Embedding Dim | Speed | Quality |
|-------|--------------|-------|---------|
| `ViT-B/32` | 512 | Fast | Good |
| `ViT-B/16` | 512 | Medium | Better |
| `ViT-L/14` | 768 | Slow | Best |
| `ViT-L/14@336px` | 768 | Very Slow | Best+ |

**Recommendation**: Start with `ViT-B/32` for fast experimentation, then use `ViT-L/14` for final results.

## Optimization Tips

- **`embedding_freq`**: Set to 4-10 for significant speedup with minimal quality loss
- **`batch_size`**: Increase to 512-1024 if you have enough GPU memory
- **`n_skills`**: Start with 10-20 for simple environments, 50+ for complex ones
- **`mem_size`**: Increase for longer training runs (but uses more RAM)

## Example Workflows

### Quick experimentation:
```bash
python main.py --config configs/halfcheetah_vit_b32.yaml --max_n_episodes 1000
```

### Full training run:
```bash
python main.py --config configs/halfcheetah_vit_l14.yaml --use_wandb
```

### Ablation study (different skill counts):
```bash
for n in 10 20 50; do
    python main.py --config configs/base.yaml --n_skills $n --use_wandb
done
```

### Multi-GPU parallel training:
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/halfcheetah_vit_b32.yaml --seed 123 &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/halfcheetah_vit_b32.yaml --seed 456 &

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/halfcheetah_vit_b32.yaml --seed 789 &
```
