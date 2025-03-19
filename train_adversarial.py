#!/usr/bin/env python
"""
Train YOLOv5 model with adversarial examples.
This script creates an adversarially patched dataset and then trains a YOLOv5 model on it.
"""

import argparse
import os
import os.path as osp
import sys
import subprocess
import torch
from pathlib import Path

from adv_patch_gen.apply_patch_to_dataset import apply_patch_to_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5 with adversarial examples")
    
    # Patch application arguments
    parser.add_argument('--patch', type=str, default="/home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png",
                        help='Path to the adversarial patch')
    parser.add_argument('--dataset', type=str, default="/home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning",
                        help='Path to the YOLO dataset directory')
    parser.add_argument('--patched-dataset', type=str, help='Output directory for patched dataset (defaults to {dataset}_patched)')
    parser.add_argument('--patch-size', type=int, nargs=2, default=(300, 300), help='Patch size (width height)')
    parser.add_argument('--target-size-frac', type=float, default=0.3, help='Fraction of object size for patch')
    parser.add_argument('--patch-alpha', type=float, default=1.0, help='Alpha for patch blending (1.0 = no blend)')
    parser.add_argument('--image-size', type=int, nargs=2, default=(640, 640), help='Image size (width height)')
    parser.add_argument('--filter-classes', type=int, nargs='+', help='Only apply patches to these class IDs')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable patch augmentation')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid'], 
                        help='Dataset splits to apply patches to (default: train and valid)')
    
    # Training arguments
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='Model configuration file')
    parser.add_argument('--data', type=str, help='Dataset configuration .yaml file (will be created if not provided)')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', 
                        help='Hyperparameters file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Train image size')
    parser.add_argument('--device', type=str, default='', help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Save to project/name')
    parser.add_argument('--name', type=str, default='adversarial_exp', help='Save to project/name')
    parser.add_argument('--workers', type=int, default=8, help='Maximum number of dataloader workers')
    
    # Mixed training arguments
    parser.add_argument('--mixed-training', action='store_true', 
                        help='Train on a mix of original and patched data (create a combined dataset)')
    parser.add_argument('--original-ratio', type=float, default=0.5, 
                        help='Ratio of original (non-patched) images to use when doing mixed training (0-1)')
    
    # Control
    parser.add_argument('--skip-patching', action='store_true', help='Skip dataset patching (use existing patched dataset)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (only create patched dataset)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (do not save any files)')
    
    return parser.parse_args()


def create_data_yaml(dataset_dir, output_path, class_file=None, dry_run=False):
    """Create a data.yaml file for YOLOv5 training."""
    # Default to looking for labels.txt in the dataset directory
    if class_file is None:
        class_file = osp.join(dataset_dir, 'labels.txt')
    
    # Read class names
    if osp.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Default class names if file not found
        print(f"Warning: Class file {class_file} not found, using default class names")
        class_names = ['class0', 'class1', 'class2']
    
    # Set up paths
    train_path = osp.join(dataset_dir, 'train')
    val_path = osp.join(dataset_dir, 'valid')
    test_path = osp.join(dataset_dir, 'test')
    
    # Create data.yaml content
    yaml_content = f"""# YOLOv5 dataset config
# Train/val/test sets

path: {dataset_dir}  # dataset root dir
train: {train_path}/images  # train images
val: {val_path}/images  # val images
test: {test_path}/images  # test images (optional)

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""

    # Write to file if not in dry_run mode
    if not dry_run:
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created dataset config at {output_path}")
    else:
        print(f"Dry run - would create dataset config at {output_path}")
    
    return output_path


def run_training(args, dataset_path):
    """Run YOLOv5 training with the specified arguments."""
    # Format device properly if it's just a number
    device = args.device
    if device and device.isdigit():
        device = f"cuda:{device}"
    
    # In dry-run mode, just show what would be done without actually running the training
    if args.dry_run:
        print("Dry run - would train with the following settings:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Weights: {args.weights}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Image size: {args.img_size}")
        print(f"  Device: {device if device else 'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("Dry run - skipping actual training")
        return
        
    # Ensure data yaml exists
    if args.data is None:
        data_yaml = osp.join(dataset_path, 'data.yaml')
        if not osp.exists(data_yaml):
            data_yaml = create_data_yaml(dataset_path, data_yaml, dry_run=args.dry_run)
    else:
        data_yaml = args.data
    
    # Check if validation directory exists, if not, use training data for validation
    valid_images_dir = osp.join(dataset_path, 'valid', 'images')
    if not osp.exists(valid_images_dir):
        print(f"Warning: Validation directory not found at {valid_images_dir}")
        
        # If the train directory exists, create a symbolic link from train to valid
        train_images_dir = osp.join(dataset_path, 'train', 'images')
        train_labels_dir = osp.join(dataset_path, 'train', 'labels')
        
        if osp.exists(train_images_dir):
            print("Creating valid directory with symlinks to train data")
            
            # Create valid directory structure
            valid_dir = osp.join(dataset_path, 'valid')
            os.makedirs(valid_dir, exist_ok=True)
            
            # Create symlinks for images and labels
            os.makedirs(valid_images_dir, exist_ok=True)
            valid_labels_dir = osp.join(dataset_path, 'valid', 'labels')
            os.makedirs(valid_labels_dir, exist_ok=True)
            
            # Copy a few images and labels for validation (10% of training data)
            print("Copying a subset of training data to validation directory")
            import glob
            import shutil
            import random
            
            # Get list of training images
            train_images = glob.glob(osp.join(train_images_dir, "*.*"))
            
            # Select 10% or at least 10 images for validation
            val_count = max(10, int(len(train_images) * 0.1))
            val_images = random.sample(train_images, min(val_count, len(train_images)))
            
            # Copy selected images and their corresponding labels
            for img_path in val_images:
                # Copy image
                img_name = osp.basename(img_path)
                shutil.copy2(img_path, osp.join(valid_images_dir, img_name))
                
                # Copy corresponding label if it exists
                label_name = osp.splitext(img_name)[0] + '.txt'
                label_path = osp.join(train_labels_dir, label_name)
                if osp.exists(label_path):
                    shutil.copy2(label_path, osp.join(valid_labels_dir, label_name))
            
            print(f"Copied {len(val_images)} images to validation directory")
    
    # Apply PIL Image ANTIALIAS fix for newer Pillow versions
    try:
        import pillow_fix
    except ImportError:
        print("Warning: pillow_fix.py not found. If you encounter PIL.Image.ANTIALIAS errors, create this file.")
    
    # Prepare training command
    cmd = [
        'python', '-c', 
        'import pillow_fix; import sys; import runpy; sys.argv = ["train.py"] + sys.argv[1:]; runpy.run_path("train.py", run_name="__main__")',
        '--weights', args.weights,
        '--data', data_yaml,
        '--hyp', args.hyp,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--img', str(args.img_size),
        '--project', args.project,
        '--name', args.name,
        '--workers', str(args.workers),
    ]
    
    # Add optional arguments
    if args.cfg:
        cmd.extend(['--cfg', args.cfg])
    if args.device:
        cmd.extend(['--device', device])
    
    # Run the training command
    print(f"Running training with command: {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    else:
        print("Dry run - skipping actual training")


def main():
    args = parse_args()
    
    # Set default patched dataset path if not provided
    if args.patched_dataset is None:
        args.patched_dataset = f"{args.dataset}_patched"
    
    # Properly format device string if it's just a number
    device = args.device
    if device and device.isdigit():
        device = f"cuda:{device}"
    else:
        device = args.device
    
    # Step 1: Apply patches to dataset (if not skipped)
    if not args.skip_patching:
        print(f"Creating adversarially patched dataset at: {args.patched_dataset}")
        apply_patch_to_dataset(
            patch_path=args.patch,
            dataset_dir=args.dataset,
            output_dir=args.patched_dataset,
            patch_size=tuple(args.patch_size),
            target_size_frac=args.target_size_frac,
            patch_alpha=args.patch_alpha,
            image_size=tuple(args.image_size),
            device=device if device else ('cuda' if torch.cuda.is_available() else 'cpu'),
            filter_class_ids=args.filter_classes,
            batch_size=args.batch_size,
            use_augmentation=not args.no_augmentation,
            apply_to_splits=args.splits,
            dry_run=args.dry_run,
        )
        
        # Copy labels.txt from original dataset if it exists
        if not args.dry_run:
            orig_labels_file = osp.join(args.dataset, 'labels.txt')
            patched_labels_file = osp.join(args.patched_dataset, 'labels.txt')
            
            if osp.exists(orig_labels_file):
                print(f"Copying labels.txt from {orig_labels_file} to {patched_labels_file}")
                import shutil
                shutil.copy2(orig_labels_file, patched_labels_file)
    
    # Skip training if requested
    if args.skip_training:
        print("Skipping training as --skip-training was specified")
        return
    
    # Step 2: Run training
    if args.mixed_training and not args.dry_run:
        # TODO: Implement mixed training by creating a combined dataset
        print("Mixed training not yet implemented")
        sys.exit(1)
    
    # For dry-run, use original dataset path for information display
    training_dataset = args.dataset if args.dry_run else args.patched_dataset
    
    # Update args.device with the properly formatted device string
    args.device = device
    
    # Run regular training on patched dataset
    run_training(args, training_dataset)


if __name__ == "__main__":
    main() 