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