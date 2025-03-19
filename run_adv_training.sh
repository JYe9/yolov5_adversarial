#!/bin/bash
# Run adversarial training with rectangular image handling

# Stop on errors
set -e

# Set output directory name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_NAME="adversarial_${TIMESTAMP}"

# Parameters
PATCH_PATH="/home/jye00001/fsdata/yolov5_adversarial/data/patch/e_100.png"
DATASET_PATH="/home/jye00001/fsdata/yolov5_adversarial/data/datasets/Adversarial_Trainning"
PATCHED_DATASET_PATH="${DATASET_PATH}_patched"
GPU_ID=3
BATCH_SIZE=16
EPOCHS=10

echo "=== Starting Adversarial Training ==="
echo "Using patch: ${PATCH_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output name: ${OUTPUT_NAME}"
echo "GPU: ${GPU_ID}"

# Check if patch exists
if [ ! -f "${PATCH_PATH}" ]; then
    echo "Error: Patch file not found at ${PATCH_PATH}"
    exit 1
fi

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "Error: Dataset directory not found at ${DATASET_PATH}"
    exit 1
fi

# Clean up previous patched dataset if it exists
if [ -d "${PATCHED_DATASET_PATH}" ]; then
    echo "Removing existing patched dataset at ${PATCHED_DATASET_PATH}"
    rm -rf "${PATCHED_DATASET_PATH}"
fi

# Ensure PIL fix is in place
if [ ! -f "pillow_fix.py" ]; then
    echo "Creating PIL fix file for ANTIALIAS compatibility"
    cat > pillow_fix.py << 'EOF'
#!/usr/bin/env python
"""
Fix for PIL.Image.ANTIALIAS deprecation in newer Pillow versions.
This file applies a monkey patch to ensure compatibility with YOLOv5's TensorBoard code.
"""

import PIL
from PIL import Image

# Fix for PIL.Image.ANTIALIAS deprecation in Pillow 10.0.0+
if not hasattr(Image, 'ANTIALIAS'):
    print("Applying PIL.Image.ANTIALIAS patch for compatibility with YOLOv5")
    # Use LANCZOS as replacement for ANTIALIAS
    Image.ANTIALIAS = Image.LANCZOS

print(f"Using PIL/Pillow version: {PIL.__version__}")
EOF
    chmod +x pillow_fix.py
fi

# Run the training
python train_adversarial.py \
    --patch "${PATCH_PATH}" \
    --dataset "${DATASET_PATH}" \
    --patched-dataset "${PATCHED_DATASET_PATH}" \
    --device "${GPU_ID}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --name "${OUTPUT_NAME}" \
    --splits train valid

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "=== Training complete ==="
    echo "Results saved to: runs/train/${OUTPUT_NAME}"
else
    echo "=== Training failed ==="
    echo "Please check the error messages above."
fi 