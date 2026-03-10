# Training Configuration Examples

## Quick Sample Run (sanity check)
```bash
python train.py --sample
```

## Basic Training (CPU, pretrained ResNet18)
```bash
python train.py --pretrained --use_sampler
```

## Full Training with Custom Parameters
```bash
python train.py \
  --model resnet34 \
  --pretrained \
  --img_size 224 \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --train_split 0.7 \
  --val_split 0.15 \
  --use_sampler \
  --out_dir reports
```

## GPU Training (if available)
```bash
python train.py --pretrained --use_sampler --num_workers 4
```

## Parameters:
- `--data_root`: Path to Data folder (default: "Data")
- `--out_dir`: Output directory for results (default: "reports")
- `--model`: Model architecture [resnet18, resnet34, resnet50] (default: resnet18)
- `--pretrained`: Use ImageNet pretrained weights
- `--img_size`: Input image size (default: 224)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 12)
- `--lr`: Learning rate (default: 2e-4)
- `--train_split`: Train split ratio (default: 0.7)
- `--val_split`: Validation split ratio (default: 0.15)
- `--use_sampler`: Enable weighted sampling for class imbalance
- `--num_workers`: DataLoader workers (default: 0 for Windows)
- `--cpu`: Force CPU usage
- `--seed`: Random seed (default: 42)
- `--sample`: Run sample check only (no training)
