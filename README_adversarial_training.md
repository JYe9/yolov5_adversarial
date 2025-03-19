# YOLOv5 Adversarial Training

This repository contains scripts for applying adversarial patches to a YOLO dataset and training a robust YOLOv5 model using adversarial training.

## Overview

Adversarial training is a defensive technique that trains models on adversarial examples to make them more robust against adversarial attacks. This repository provides tools to:

1. Generate adversarial patches using gradient-based optimization
2. Apply adversarial patches to a YOLO dataset
3. Train a YOLOv5 model on the adversarially patched dataset

## What Are Adversarial Patches?

Adversarial patches are specially crafted images designed to fool object detectors when placed in a scene. Unlike traditional adversarial examples that require modifying the entire image, patches can be printed and placed in the physical world to attack detectors in real-time.

![Adversarial Patch Example](data/patch/e_100.png)

In simple terms:
- **Normal image**: The detector correctly identifies objects
- **Image with adversarial patch**: The detector fails to identify objects or misclassifies them

Our implementation follows the equation:
```
y' = patch + (x, y, w, h)
```
Where:
- `y'` is the modified image with the patch
- `patch` is the adversarial patch
- `(x, y, w, h)` are the coordinates from the YOLO labels that specify where to apply the patch

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

3. Fix for Pillow compatibility issues:
   - If you encounter `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` error, we've created a fix:
   ```bash
   # The pillow_fix.py module is automatically imported in train_adversarial.py
   # It handles the ANTIALIAS deprecation by using Image.LANCZOS instead
   ```

## File Structure

- `adv_patch_gen/apply_patch_to_dataset.py`: Script to apply adversarial patches to a dataset
- `adv_patch_gen/utils/patch.py`: Contains PatchTransformer and PatchApplier classes
- `train_patch.py`: Script to generate adversarial patches
- `train_adversarial.py`: Script to create a patched dataset and train a YOLOv5 model
- `test_patch.py`: Script to test adversarial patches on images
- `run_adv_training.sh`: Convenient shell script to run the full adversarial training pipeline
- `pillow_fix.py`: Module that fixes Pillow compatibility issues

## Detailed Implementation

### Key Components

1. **PatchTransformer** (in `adv_patch_gen/utils/patch.py`):
   
   This class handles the transformation of the patch based on the label coordinates:
   
   ```python
   # Key transformation steps
   # 1. Scale the patch size based on bounding box dimensions
   tsize = np.random.uniform(*self.t_size_frac)
   w_size = lab_batch_scaled[:, :, 3].mul(tsize)  # width
   h_size = lab_batch_scaled[:, :, 4].mul(tsize)  # height
   target_size = torch.sqrt(w_size**2 + h_size**2)
   
   # 2. Get target position from labels
   target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # x center
   target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # y center
   ```

2. **PatchApplier** (in `adv_patch_gen/utils/patch.py`):
   
   This class handles the actual application of the patch to the image:
   
   ```python
   def forward(self, img_batch, adv_batch):
       advs = torch.unbind(adv_batch, 1)
       for adv in advs:
           if self.patch_alpha == 1:
               # Direct replacement
               img_batch = torch.where((adv == 0), img_batch, adv)
           else:
               # Alpha blending
               alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
               img_batch = torch.where((adv == 0), img_batch, alpha_blend)
       return img_batch
   ```

3. **Apply Patch to Dataset** (in `adv_patch_gen/apply_patch_to_dataset.py`):
   
   The main function that processes images and applies patches:
   
   ```python
   # Apply patch
   with torch.no_grad():
       adv_batch_t = patch_transformer(
           patch_tensor,
           label_tensor,
           image_size,
           use_mul_add_gau=use_augmentation,
           do_transforms=use_augmentation,
           do_rotate=use_augmentation,
           rand_loc=False  # Use exact label positions
       )
       patched_img_tensor = patch_applier(img_tensor, adv_batch_t)
   ```

## Step-by-Step Guide

### Option 1: Using the Shell Script (Recommended)

We've created a convenient shell script that automates the entire process:

```bash
./run_adv_training.sh
```

This script:
1. Checks for the existence of required files and directories
2. Creates a timestamped output directory
3. Cleans up any existing patched datasets
4. Applies the adversarial patch to your dataset
5. Trains the YOLOv5 model on the patched dataset

### Option 2: Complete Adversarial Training Pipeline

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

### Option 3: Create Patched Dataset Only

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

### Option 4: Training Only (Using Pre-Patched Dataset)

If you already have a patched dataset and want to train on it:

```bash
python train_adversarial.py \
    --patched-dataset /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched \
    --skip-patching \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16
```

## Recent Fixes and Improvements

Our implementation has been enhanced with several important fixes:

1. **Pillow Compatibility Fix**:
   - Added `pillow_fix.py` to handle the deprecated `ANTIALIAS` attribute in newer Pillow versions
   - Code snippet:
   ```python
   import PIL
   from PIL import Image

   # Check if ANTIALIAS is missing (in newer Pillow versions)
   if not hasattr(Image, 'ANTIALIAS'):
       # Use LANCZOS as the replacement
       Image.ANTIALIAS = Image.LANCZOS
   ```

2. **Special Characters in Filenames**:
   - Fixed errors when processing files with spaces or special characters in their names
   - Code snippet:
   ```python
   try:
       patched_img.save(out_img_path, quality=95)
   except Exception as e:
       # Try saving with a sanitized filename
       sanitized_base_name = base_name.replace(' ', '_').replace('(', '').replace(')', '')
       alt_img_path = osp.join(output_dir, split, 'images', f"{sanitized_base_name}.jpg")
       patched_img.save(alt_img_path, quality=95)
   ```

3. **Validation Dataset Creation**:
   - Added automatic creation of validation dataset if missing
   - The script now copies a portion of training images for validation when needed

4. **Robust Shell Script**:
   - Added error handling and validation checks
   - Created timestamped output directories for better experiment tracking
   - Implemented cleanup for existing patched datasets

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

## Visualization

To visualize the adversarial patch application:

1. Enable debug mode in `train_patch.py`:
   ```python
   cfg.debug_mode = True
   ```

2. The patched images with bounding boxes will be saved in:
   ```
   {log_dir}/val_patch_applied_imgs/
   ```

This allows you to see how the patches are being applied to the objects in your dataset.

## Troubleshooting

### Common Issues and Solutions

1. **Error: No module named 'PIL.Image' has no attribute 'ANTIALIAS'**
   - **Solution**: Make sure `pillow_fix.py` is imported at the beginning of your script
   - Alternatively, upgrade or downgrade Pillow: `pip install pillow==9.0.0`

2. **Error: FileNotFoundError for image files with spaces or special characters**
   - **Solution**: Our latest code automatically handles this by sanitizing filenames

3. **Error: CUDA out of memory**
   - **Solution**: Reduce batch size with the `--batch-size` parameter or use a smaller model

4. **Error: No labels found for certain images**
   - **Solution**: Check your dataset structure. Images should be in `{split}/images/` and labels in `{split}/labels/`

## Advanced Usage

### Creating Your Own Adversarial Patches

To generate your own adversarial patches:

```bash
python train_patch.py \
    --config configs/patch/example.yaml
```

The configuration file contains all the parameters for patch generation, including:
- Target class to attack
- Learning rate and optimization parameters
- Patch dimensions and constraints

### Testing Patches on Images

To test the effectiveness of your adversarial patches:

```bash
python test_patch.py \
    --patch data/patch/your_custom_patch.png \
    --img-dir data/test_images \
    --output-dir results/patch_tests
```

## How Adversarial Training Makes Models More Robust

In simple terms:
1. **Normal training**: Model learns to detect objects in clean images
2. **Adversarial training**: Model learns to detect objects even when adversarial patches are present

By exposing the model to adversarial examples during training, it becomes less susceptible to these attacks during deployment. Think of it as "immunizing" the model against potential attacks.

## Example Training Log
```bash
(yolov5-gpu) [jye00001@kiosgpusim3 yolov5_adversarial]$ ./run_adv_training.sh 
=== Starting Adversarial Training ===
Using patch: /home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png
Dataset: /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning
Output name: adversarial_20250319_145213
GPU: 3
Removing existing patched dataset at /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched
Creating adversarially patched dataset at: /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched
Processing 636 images in train split
Applying patches to train: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 636/636 [02:01<00:00,  5.25it/s]
Processing 154 images in valid split
Applying patches to valid: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 154/154 [00:21<00:00,  7.27it/s]
Processing complete!
Patched dataset saved to: /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched
Copying labels.txt from /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning/labels.txt to /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/labels.txt
Created dataset config at /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/data.yaml
Applying PIL.Image.ANTIALIAS patch for compatibility with YOLOv5
Using PIL/Pillow version: 11.1.0
Running training with command: python -c import pillow_fix; import sys; import runpy; sys.argv = ["train.py"] + sys.argv[1:]; runpy.run_path("train.py", run_name="__main__") --weights yolov5s.pt --data /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/data.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 10 --batch-size 16 --img 640 --project runs/train --name adversarial_20250319_145213 --workers 8 --device cuda:3
Applying PIL.Image.ANTIALIAS patch for compatibility with YOLOv5
Using PIL/Pillow version: 11.1.0
wandb: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
wandb: Currently logged in as: jye2024. Use `wandb login --relogin` to force relogin
train: weights=yolov5s.pt, cfg=, data=/home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=10, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data/hyps, resume_evolve=None, bucket=, cache=None, image_weights=False, device=cuda:3, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=adversarial_20250319_145213, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, patch_dir=, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
github: ⚠️ YOLOv5 is out of date by 10 commits. Use 'git pull ultralytics master' or 'git clone https://github.com/ultralytics/yolov5' to update.
YOLOv5 🚀 v7.0-536-g901a3569 Python-3.10.9 torch-1.12.1 CUDA:3 (Tesla V100-SXM2-32GB, 32494MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, bbox_patch=0.3
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: wandb version 0.19.8 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.13.10
wandb: Run data is saved locally in /home/jye00001/fsdata/yolov5_adversarial/wandb/run-20250319_145445-vcxzmorc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run adversarial_20250319_145213
wandb: ⭐️ View project at https://wandb.ai/jye2024/YOLOv5
wandb: 🚀 View run at https://wandb.ai/jye2024/YOLOv5/runs/vcxzmorc
Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     21576  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
train: Scanning /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/train/labels... 636 images, 0 backgrounds, 0 corrupt: 100%|██████████| 636/6
train: WARNING ⚠️ /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/train/images/DJI_0955 (6_23)_n0.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/train/images/lim_3630.jpg: 1 duplicate labels removed
train: New cache created: /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/train/labels.cache
val: Scanning /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/valid/labels... 154 images, 0 backgrounds, 0 corrupt: 100%|██████████| 154/154
val: New cache created: /home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning_patched/valid/labels.cache

AutoAnchor: 3.09 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/adversarial_20250319_145213/labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/adversarial_20250319_145213
Starting training for 10 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        0/9      3.35G     0.1291    0.06547    0.03042        593        640: 100%|██████████| 40/40 [00:06<00:00,  6.40it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:01<00:00,  3.53it/s]
                   all        154       2831    0.00335     0.0692    0.00252   0.000539

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        1/9      4.04G     0.1088    0.05911    0.01701        417        640: 100%|██████████| 40/40 [00:03<00:00, 10.36it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:01<00:00,  3.81it/s]
                   all        154       2831       0.69      0.105     0.0142    0.00278

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        2/9      4.04G    0.09295    0.06453    0.01475        654        640: 100%|██████████| 40/40 [00:04<00:00,  8.78it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:01<00:00,  4.66it/s]
                   all        154       2831      0.688      0.162     0.0139    0.00337

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        3/9      4.04G    0.08574    0.06531    0.01441        811        640: 100%|██████████| 40/40 [00:03<00:00, 10.70it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:00<00:00,  5.16it/s]
                   all        154       2831      0.734      0.249     0.0718     0.0211

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        4/9      4.04G    0.07386    0.06216    0.01331        265        640: 100%|██████████| 40/40 [00:03<00:00, 10.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:01<00:00,  4.92it/s]
                   all        154       2831      0.786      0.222      0.166     0.0391

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        5/9      4.04G    0.06617    0.06318    0.01342        551        640: 100%|██████████| 40/40 [00:03<00:00, 10.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:00<00:00,  5.25it/s]
                   all        154       2831      0.869      0.295      0.272     0.0821

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        6/9      4.04G    0.05942    0.06391    0.01428        634        640: 100%|██████████| 40/40 [00:03<00:00, 10.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:00<00:00,  5.27it/s]
                   all        154       2831      0.911      0.292      0.314      0.106

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        7/9      4.04G    0.05668    0.06202    0.01287        443        640: 100%|██████████| 40/40 [00:03<00:00, 10.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:00<00:00,  5.11it/s]
                   all        154       2831      0.958      0.316      0.362      0.142

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        8/9      4.04G    0.05098    0.06067    0.01298        393        640: 100%|██████████| 40/40 [00:03<00:00, 10.72it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:00<00:00,  5.21it/s]
                   all        154       2831      0.961      0.317        0.4      0.202

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        9/9      4.04G    0.04723    0.05883     0.0127        352        640: 100%|██████████| 40/40 [00:03<00:00, 10.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:00<00:00,  5.19it/s]
                   all        154       2831      0.974      0.324       0.48      0.267

10 epochs completed in 0.019 hours.
Optimizer stripped from runs/train/adversarial_20250319_145213/weights/last.pt, 14.5MB
Optimizer stripped from runs/train/adversarial_20250319_145213/weights/best.pt, 14.5MB

Validating runs/train/adversarial_20250319_145213/weights/best.pt...
Fusing layers... 
Model summary: 157 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5/5 [00:02<00:00,  2.29it/s]
                   all        154       2831      0.974      0.324       0.48      0.268
                   car        154       2721      0.921      0.971      0.944      0.489
                   bus        154         43          1          0      0.442      0.282
                 truck        154         67          1          0     0.0541     0.0327
Results saved to runs/train/adversarial_20250319_145213
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▁▁▂▃▅▆▆▇██
wandb: metrics/mAP_0.5:0.95 ▁▁▁▂▂▃▄▅▆██
wandb:    metrics/precision ▁▆▆▆▇▇█████
wandb:       metrics/recall ▁▂▄▆▅▇▇████
wandb:       train/box_loss █▆▅▄▃▃▂▂▁▁
wandb:       train/cls_loss █▃▂▂▁▁▂▁▁▁
wandb:       train/obj_loss █▁▇█▅▆▆▄▃▁
wandb:         val/box_loss █▇▆▅▄▃▃▂▂▁▁
wandb:         val/cls_loss █▂▁▂▁▁▂▁▁▁▁
wandb:         val/obj_loss ▃█▃▃▃▂▂▂▁▁▁
wandb:                x/lr0 █▅▂▂▂▁▁▁▁▁
wandb:                x/lr1 ▂▆█▇▇▆▅▃▂▁
wandb:                x/lr2 ▂▆█▇▇▆▅▃▂▁
wandb: 
wandb: Run summary:
wandb:           best/epoch 9
wandb:         best/mAP_0.5 0.48016
wandb:    best/mAP_0.5:0.95 0.26689
wandb:       best/precision 0.97353
wandb:          best/recall 0.32402
wandb:      metrics/mAP_0.5 0.48025
wandb: metrics/mAP_0.5:0.95 0.26788
wandb:    metrics/precision 0.97351
wandb:       metrics/recall 0.32378
wandb:       train/box_loss 0.04723
wandb:       train/cls_loss 0.0127
wandb:       train/obj_loss 0.05883
wandb:         val/box_loss 0.03092
wandb:         val/cls_loss 0.00839
wandb:         val/obj_loss 0.02721
wandb:                x/lr0 0.00208
wandb:                x/lr1 0.00208
wandb:                x/lr2 0.00208
wandb: 
wandb: 🚀 View run adversarial_20250319_145213 at: https://wandb.ai/jye2024/YOLOv5/runs/vcxzmorc
wandb: Synced 6 W&B file(s), 17 media file(s), 3 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250319_145445-vcxzmorc/logs
wandb: WARNING ⚠️ wandb is deprecated and will be removed in a future release. See supported integrations at https://github.com/ultralytics/yolov5#integrations.
=== Training complete ===
Results saved to: runs/train/adversarial_20250319_145213
```


## Notice

Training on adversarial examples can make models more robust against the specific attack patterns used during training, but may not guarantee robustness against all possible adversarial attacks. It's recommended to combine this approach with other security measures for critical applications.

## References

- YOLOv5: https://github.com/ultralytics/yolov5
- Original Adversarial Patch Repository: https://github.com/SamSamhuns/yolov5_adversarial 