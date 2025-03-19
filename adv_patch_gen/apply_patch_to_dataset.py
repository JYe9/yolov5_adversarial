#!/usr/bin/env python
"""
Apply adversarial patches to a dataset based on label positions.
This script processes images and applies adversarial patches at the positions of labeled objects.
"""

import argparse
import glob
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from adv_patch_gen.utils.common import IMG_EXTNS, pad_to_square
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer


class YOLOLabeledImage:
    """Class to represent a YOLO labeled image."""
    
    def __init__(self, img_path: str, label_path: str = None):
        """Initialize with image and optional label path."""
        self.img_path = img_path
        self.label_path = label_path
        
        # Extract base filename without extension
        self.base_name = osp.splitext(osp.basename(img_path))[0]
        
        # Try to find label file if not provided
        if label_path is None:
            img_dir = osp.dirname(img_path)
            label_dir = osp.join(osp.dirname(img_dir), 'labels')
            potential_label = osp.join(label_dir, f"{self.base_name}.txt")
            if osp.exists(potential_label):
                self.label_path = potential_label
    
    def load_image(self, mode: str = "RGB") -> Image.Image:
        """Load the image as PIL Image."""
        return Image.open(self.img_path).convert(mode)
    
    def load_labels(self) -> np.ndarray:
        """Load YOLO format labels."""
        if self.label_path is None or not osp.exists(self.label_path):
            return np.zeros((1, 5))  # Return dummy label if no label file
        
        try:
            with open(self.label_path, 'r') as f:
                labels = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Format: class_id, x_center, y_center, width, height
                        labels.append([float(x) for x in parts[:5]])
                
                return np.array(labels) if labels else np.zeros((1, 5))
        except Exception as e:
            print(f"Error loading labels for {self.label_path}: {e}")
            return np.zeros((1, 5))


def apply_patch_to_dataset(
    patch_path: str,
    dataset_dir: str,
    output_dir: str = None,
    patch_size: Tuple[int, int] = (300, 300),
    target_size_frac: float = 0.3,
    patch_alpha: float = 1.0,
    image_size: Tuple[int, int] = (640, 640),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    filter_class_ids: List[int] = None,
    batch_size: int = 16,
    use_augmentation: bool = True,
    apply_to_splits: List[str] = ['train', 'valid'],
    dry_run: bool = False,
):
    """
    Apply adversarial patch to dataset images based on label positions.
    
    Args:
        patch_path: Path to the adversarial patch image
        dataset_dir: Path to the YOLO dataset directory
        output_dir: Directory to save patched images (defaults to {dataset_dir}_patched)
        patch_size: Size of the patch as (width, height)
        target_size_frac: Fraction of object size for the patch
        patch_alpha: Alpha blending factor for patch application (1.0 = no blending)
        image_size: Size to resize images to before patch application
        device: Device to use for processing ('cuda' or 'cpu')
        filter_class_ids: Only apply patches to these class IDs (None = all classes)
        batch_size: Batch size for processing
        use_augmentation: Whether to use augmentation on the patches
        apply_to_splits: Which dataset splits to process ('train', 'valid', 'test')
        dry_run: If True, only process a few images without saving
    """
    if dry_run:
        print("Running in dry-run mode, no files will be saved")
    
    # Set device
    dev = torch.device(device)
    
    # Create output directory
    if not dry_run:
        if output_dir is None:
            output_dir = f"{dataset_dir}_patched"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directories for each split
        for split in apply_to_splits:
            split_img_dir = osp.join(output_dir, split, 'images')
            split_label_dir = osp.join(output_dir, split, 'labels')
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_label_dir, exist_ok=True)
    
    # Load the adversarial patch
    patch_img = Image.open(patch_path)
    patch_img = patch_img.resize(patch_size)
    patch_tensor = T.ToTensor()(patch_img).to(dev)
    
    # Initialize patch transformer and applier
    patch_transformer = PatchTransformer(
        t_size_frac=target_size_frac,
        mul_gau_mean=(0.5, 0.8),
        mul_gau_std=0.1,
        x_off_loc=[-0.25, 0.25],
        y_off_loc=[-0.25, 0.25],
        dev=dev
    ).to(dev)
    
    patch_applier = PatchApplier(patch_alpha=patch_alpha, dev=dev).to(dev)
    
    # Process each split
    for split in apply_to_splits:
        split_img_dir = osp.join(dataset_dir, split, 'images')
        split_label_dir = osp.join(dataset_dir, split, 'labels')
        
        # Check if directories exist
        if not osp.exists(split_img_dir):
            print(f"Warning: {split_img_dir} does not exist, skipping")
            continue
            
        # Get all image paths
        img_paths = []
        for ext in IMG_EXTNS:
            img_paths.extend(glob.glob(osp.join(split_img_dir, f"*{ext}")))
        
        img_paths = sorted(img_paths)
        
        if dry_run:
            # Limit to first 5 images for dry run
            img_paths = img_paths[:5]
        
        print(f"Processing {len(img_paths)} images in {split} split")
        
        # Process images
        for img_path in tqdm(img_paths, desc=f"Applying patches to {split}"):
            # Get corresponding label path
            base_name = osp.splitext(osp.basename(img_path))[0]
            label_path = osp.join(split_label_dir, f"{base_name}.txt")
            
            # Skip if label doesn't exist
            if not osp.exists(label_path):
                print(f"Warning: No label found for {img_path}, skipping")
                continue
            
            # Load image and label
            yolo_img = YOLOLabeledImage(img_path, label_path)
            img = yolo_img.load_image()
            labels = yolo_img.load_labels()
            
            # Filter labels by class ID if requested
            if filter_class_ids is not None:
                labels = np.array([label for label in labels if int(label[0]) in filter_class_ids])
                if len(labels) == 0:
                    # Skip images with no matching classes
                    continue
            
            # Get original image dimensions
            orig_width, orig_height = img.size
            
            # Prepare image for processing - resize without padding
            img_resized = img.resize(image_size)
            img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(dev)
            
            # Adjust labels for resized dimensions
            scaled_labels = labels.copy()
            # We don't need to adjust x,y,w,h as they are normalized to 0-1 already in YOLO format
            
            # Convert labels to tensor format (need at least one label)
            if len(scaled_labels) == 0:
                scaled_labels = np.zeros((1, 5))
            label_tensor = torch.from_numpy(scaled_labels).float().unsqueeze(0).to(dev)
            
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
            
            # Convert back to PIL and save
            patched_img = T.ToPILImage()(patched_img_tensor.squeeze(0).cpu())
            
            if not dry_run:
                # Save patched image and copy label
                out_img_path = osp.join(output_dir, split, 'images', f"{base_name}.jpg")
                out_label_path = osp.join(output_dir, split, 'labels', f"{base_name}.txt")
                
                # Ensure the directory exists (handle special characters in paths)
                os.makedirs(osp.dirname(out_img_path), exist_ok=True)
                os.makedirs(osp.dirname(out_label_path), exist_ok=True)
                
                # Save the image, handling potential errors
                try:
                    patched_img.save(out_img_path, quality=95)
                except Exception as e:
                    print(f"Error saving image to {out_img_path}: {e}")
                    # Try saving with a sanitized filename
                    sanitized_base_name = base_name.replace(' ', '_').replace('(', '').replace(')', '')
                    alt_img_path = osp.join(output_dir, split, 'images', f"{sanitized_base_name}.jpg")
                    alt_label_path = osp.join(output_dir, split, 'labels', f"{sanitized_base_name}.txt")
                    print(f"Attempting to save with sanitized filename: {alt_img_path}")
                    patched_img.save(alt_img_path, quality=95)
                    
                    # Use the sanitized filename for the label too
                    with open(label_path, 'r') as src, open(alt_label_path, 'w') as dst:
                        dst.write(src.read())
                    continue
                
                # Copy label file
                try:
                    with open(label_path, 'r') as src, open(out_label_path, 'w') as dst:
                        dst.write(src.read())
                except Exception as e:
                    print(f"Error copying label from {label_path} to {out_label_path}: {e}")
            
            if dry_run:
                # Display some debug info
                print(f"Processed: {img_path}")
                print(f"  - Found {len(labels)} labels")
                print(f"  - Original image size: {(orig_width, orig_height)}")
                print(f"  - Resized to: {image_size}")
                print(f"  - No padding used, maintaining aspect ratio")
    
    print("Processing complete!")
    if not dry_run:
        print(f"Patched dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Apply adversarial patches to a YOLO dataset")
    parser.add_argument('--patch', type=str, required=True, help='Path to the adversarial patch')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the YOLO dataset directory')
    parser.add_argument('--output', type=str, help='Output directory for patched dataset')
    parser.add_argument('--patch-size', type=int, nargs=2, default=(300, 300), help='Patch size (width height)')
    parser.add_argument('--target-size-frac', type=float, default=0.3, help='Fraction of object size for patch')
    parser.add_argument('--patch-alpha', type=float, default=1.0, help='Alpha for patch blending (1.0 = no blend)')
    parser.add_argument('--image-size', type=int, nargs=2, default=(640, 640), help='Image size (width height)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--filter-classes', type=int, nargs='+', help='Only apply patches to these class IDs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable patch augmentation')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid'], 
                        help='Dataset splits to process')
    parser.add_argument('--dry-run', action='store_true', help='Run without saving files')
    
    args = parser.parse_args()
    
    apply_patch_to_dataset(
        patch_path=args.patch,
        dataset_dir=args.dataset,
        output_dir=args.output,
        patch_size=tuple(args.patch_size),
        target_size_frac=args.target_size_frac,
        patch_alpha=args.patch_alpha,
        image_size=tuple(args.image_size),
        device=args.device,
        filter_class_ids=args.filter_classes,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
        apply_to_splits=args.splits,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main() 