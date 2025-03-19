# YOLOv5 Adversarial Training

This repository contains scripts for applying adversarial patches to a YOLO dataset and training a robust YOLOv5 model using adversarial training.

## Overview

Adversarial training is a defensive technique that trains models on adversarial examples to make them more robust against adversarial attacks. This repository provides tools to:

1. Apply adversarial patches to a YOLO dataset
2. Train a YOLOv5 model on the adversarially patched dataset

## Setup

1. Clone the YOLOv5 repository (if you haven't already):
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   ```

2. Ensure you have the necessary dependencies:
   ```bash
   pip install torch torchvision tqdm pillow easydict numpy
   ```

## File Structure

- `adv_patch_gen/apply_patch_to_dataset.py`: Script to apply adversarial patches to a dataset
- `train_adversarial.py`: Script to create a patched dataset and train a YOLOv5 model
- `data/patch/e_100.png`: Pre-generated adversarial patch

## Usage

### Option 1: Complete Adversarial Training Pipeline

To apply patches to your dataset and train a YOLOv5 model in one step:

```bash
python train_adversarial.py \
    --patch /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png \
    --dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning \
    --patched-dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched \
    --epochs 100 \
    --batch-size 16 \
    --weights yolov5s.pt
```

### Option 2: Create Patched Dataset Only

If you only want to create a patched dataset without training:

```bash
python train_adversarial.py \
    --patch /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png \
    --dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning \
    --skip-training
```

Or directly use the dataset patching script:

```bash
python -m adv_patch_gen.apply_patch_to_dataset \
    --patch /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png \
    --dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning \
    --output /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched
```

### Option 3: Training Only (Using Pre-Patched Dataset)

If you already have a patched dataset and want to train on it:

```bash
python train_adversarial.py \
    --patched-dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched \
    --skip-patching \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16
```

## Parameters

### Patch Application Parameters

- `--patch`: Path to the adversarial patch
- `--dataset`: Path to the original YOLO dataset
- `--patched-dataset`: Output directory for patched dataset
- `--patch-size`: Size of the patch (width height)
- `--target-size-frac`: Fraction of object size for patch
- `--patch-alpha`: Alpha blending factor (1.0 = no blending)
- `--filter-classes`: Only apply patches to specific class IDs
- `--splits`: Which dataset splits to process (train, valid, test)

### Training Parameters

- `--weights`: Initial weights path
- `--cfg`: Model configuration file
- `--data`: Dataset configuration file
- `--hyp`: Hyperparameters file
- `--epochs`: Number of epochs
- `--batch-size`: Batch size
- `--img-size`: Training image size
- `--device`: Device to use (cuda device or cpu)
- `--name`: Experiment name for saving results

## Examples

### Dry Run (Testing)

To test the patch application without saving any files:

```bash
python -m adv_patch_gen.apply_patch_to_dataset \
    --patch /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png \
    --dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning \
    --dry-run
```

### Apply Patches to Specific Classes

To apply patches only to objects of certain classes (e.g., classes 0 and 1):

```bash
python -m adv_patch_gen.apply_patch_to_dataset \
    --patch /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png \
    --dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning \
    --filter-classes 0 1
```

### Train with Custom Hyperparameters

```bash
python train_adversarial.py \
    --patch /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png \
    --dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning \
    --hyp data/hyps/hyp.scratch-high.yaml \
    --epochs 200 \
    --batch-size 32 \
    --weights yolov5m.pt
```

## How It Works

1. The `apply_patch_to_dataset.py` script:
   - Loads each image and its corresponding labels
   - For each labeled object, applies an adversarial patch at the object's position
   - Saves the patched images and original labels

2. The `train_adversarial.py` script:
   - Creates a patched dataset using the above process
   - Sets up YOLOv5 training configuration
   - Trains the model on the patched dataset

## Notice

Training on adversarial examples can make models more robust against the specific attack patterns used during training, but may not guarantee robustness against all possible adversarial attacks.

## References

- YOLOv5: https://github.com/ultralytics/yolov5
- Adversarial Patch Paper: https://arxiv.org/abs/1712.09665 