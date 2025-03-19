# Adversarial Patch Training Process

This document provides a detailed explanation of how adversarial patches are trained and how they work as an attack against object detection systems like YOLOv5.

## What is an Adversarial Patch?

Adversarial patches are specially crafted images designed to fool object detectors when placed in a scene. Unlike traditional adversarial examples that require modifying the entire image, patches are localized perturbations that can be physically printed and placed in the real world.

The primary goal is to create a patch that, when placed on or near an object:
- Prevents detection of the object
- Causes misclassification of the object
- Reduces the confidence score below detection threshold

## Patch Training Process Overview

The patch training process creates optimized adversarial patterns through an iterative optimization process:

1. **Patch Initialization**
2. **Training Loop** with transformations and applications
3. **Loss Calculation** using specialized loss functions
4. **Gradient-Based Optimization** to update the patch
5. **Evaluation and Validation** to measure effectiveness

## 1. Patch Initialization

The patch can be initialized in several ways:

```python
# From train_patch.py
def init_patch(self, mode="random", size=(300, 300), img_path=None):
    """Initialize adversarial patch.
    Args:
        mode: How to initialize the patch
            - random: Random uniform distribution
            - gray: Solid gray color
            - file: Initialize from image file
        size: Patch dimensions (width, height)
        img_path: Path to image file (if mode='file')
    """
    if mode == "random":
        adv_patch = torch.rand(3, *size)  # (channels, height, width)
    elif mode == "gray":
        adv_patch = torch.full((3, *size), 0.5)
    elif mode == "file":
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        adv_patch = T.ToTensor()(img)
    else:
        raise ValueError(f"Invalid initialization mode: {mode}")
    
    return adv_patch
```

After initialization, the patch is registered as a trainable parameter:

```python
# Register patch as trainable parameter
self.adv_patch_raw = nn.Parameter(patch_init.to(self.dev))

# Set up optimizer for the patch parameter
self.optimizer = optim.Adam([self.adv_patch_raw], lr=self.cfg.start_learning_rate)
```

## 2. Patch Transformation

Before a patch can be applied to an image, it needs to be transformed appropriately. The `PatchTransformer` class handles this process:

```python
class PatchTransformer(nn.Module):
    """Class that applies transformations to a patch.
    
    Handles:
    - Scaling the patch based on target object size
    - Rotation and perspective transformations
    - Gaussian noise addition
    - Proper positioning over target objects
    """
    
    def __init__(
        self,
        t_size_frac=(0.2, 0.4),
        mul_gau_mean=(0.5, 0.8),
        mul_gau_std=0.1,
        x_off_loc=[-0.25, 0.25],
        y_off_loc=[-0.25, 0.25],
        dev="cuda:0",
    ):
        """Initialize PatchTransformer."""
        super(PatchTransformer, self).__init__()
        self.t_size_frac = t_size_frac
        self.mul_gau_mean = mul_gau_mean
        self.mul_gau_std = mul_gau_std
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.dev = dev
        
        # Create rotation and noise transforms
        self.rot = RandomPerspective()
        self.add_gau = AddMultiplicativeGaussianNoise()
```

The transformation process follows these steps:

```python
def forward(self, adv_patch, lab_batch, img_size, use_mul_add_gau=True, do_transforms=True, do_rotate=True, rand_loc=True):
    """Apply transformations to patch.
    
    Args:
        adv_patch: The adversarial patch
        lab_batch: Batch of labels (class, x_center, y_center, width, height)
        img_size: Size of the target image
        use_mul_add_gau: Add Gaussian noise
        do_transforms: Apply transformations
        do_rotate: Apply rotations
        rand_loc: Use random locations (vs. fixed label positions)
    """
    # Extract batch size
    batch_size = (lab_batch.size(0), lab_batch.size(1))
    
    # 1. Scale the patch based on object size
    tsize = np.random.uniform(*self.t_size_frac)
    w_size = lab_batch[:, :, 3].mul(tsize)  # width
    h_size = lab_batch[:, :, 4].mul(tsize)  # height
    target_size = torch.sqrt(w_size**2 + h_size**2)
    
    # 2. Get target position from labels
    target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # x center
    target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # y center
    
    # 3. Apply random offset if enabled
    if rand_loc:
        off_x = w_size.view(np.prod(batch_size)) * torch.empty(w_size.view(np.prod(batch_size)).size(), device=self.dev).uniform_(*self.x_off_loc)
        target_x = target_x + off_x
        off_y = h_size.view(np.prod(batch_size)) * torch.empty(h_size.view(np.prod(batch_size)).size(), device=self.dev).uniform_(*self.y_off_loc)
        target_y = target_y + off_y
    
    # 4. Apply rotations and perspective transforms
    if do_rotate:
        adv_patch = self.rot(adv_patch)
    
    # 5. Add Gaussian noise for robustness
    if use_mul_add_gau:
        adv_patch = self.add_gau(adv_patch, self.mul_gau_mean, self.mul_gau_std)
    
    # 6. Create patch masks and transform patches
    # Initialize tensors for transformed patches
    adv_batch_t = torch.zeros((np.prod(batch_size), 3, adv_patch.size(1), adv_patch.size(2)), device=self.dev)
    
    # Process each patch in the batch
    for i in range(np.prod(batch_size)):
        # Get current patch size and position
        curr_size = target_size[i]
        curr_x = target_x[i]
        curr_y = target_y[i]
        
        # Create transformation matrix for scaling
        scale = curr_size / max(adv_patch.size(1), adv_patch.size(2))
        transform = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], device=self.dev)
        
        # Apply scaling transformation
        grid = F.affine_grid(transform.unsqueeze(0), adv_patch.unsqueeze(0).size(), align_corners=True)
        scaled_patch = F.grid_sample(adv_patch.unsqueeze(0), grid, align_corners=True)
        
        # Create mask for patch placement
        mask = torch.zeros((1, 1, img_size[0], img_size[1]), device=self.dev)
        
        # Calculate patch boundaries
        patch_h, patch_w = scaled_patch.size(2), scaled_patch.size(3)
        x1 = int(curr_x * img_size[1] - patch_w/2)
        y1 = int(curr_y * img_size[0] - patch_h/2)
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        
        # Ensure patch stays within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_size[1], x2)
        y2 = min(img_size[0], y2)
        
        # Place patch in mask
        mask[:, :, y1:y2, x1:x2] = 1
        
        # Apply patch to mask
        adv_batch_t[i] = scaled_patch.squeeze(0)
    
    # 7. Apply final transformations if enabled
    if do_transforms:
        # Add random brightness and contrast
        brightness = torch.rand(1, device=self.dev) * 0.2 + 0.9  # Random between 0.9 and 1.1
        contrast = torch.rand(1, device=self.dev) * 0.2 + 0.9   # Random between 0.9 and 1.1
        adv_batch_t = adv_batch_t * brightness
        adv_batch_t = (adv_batch_t - 0.5) * contrast + 0.5
    
    # 8. Clamp values to valid range
    adv_batch_t = torch.clamp(adv_batch_t, 0.0, 1.0)
    
    return adv_batch_t
```

## 3. Patch Application

After transformation, the `PatchApplier` class handles placing the patch on the image:

```python
class PatchApplier(nn.Module):
    """Applies an adversarial patch to images based on transformed locations."""
    
    def __init__(self, patch_alpha=1.0, dev="cuda:0"):
        """Initialize PatchApplier."""
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha
        self.dev = dev
    
    def forward(self, img_batch, adv_batch):
        """Apply the patch to the image batch.
        
        Args:
            img_batch: Batch of images
            adv_batch: Batch of transformed patches
        
        Returns:
            Tensor: Patched images
        """
        # Apply each patch in the batch
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            if self.patch_alpha == 1:
                # Direct replacement where patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, adv)
            else:
                # Alpha blending between patch and image
                alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)
        
        return img_batch
```

## 4. Training Loop

The main training loop orchestrates the entire process:

```python
def train(self):
    """Optimize a patch to generate an adversarial example."""
    # Initialize data loaders, scheduler, etc.
    # ...
    
    # Training epochs
    for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
        ep_loss = 0
        
        # Process each batch
        for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
            # Move data to device
            img_batch = img_batch.to(self.dev)
            lab_batch = lab_batch.to(self.dev)
            
            # Create patch
            adv_patch = self.adv_patch_raw.clone()
            adv_patch = torch.clamp(adv_patch, 0.0, 1.0)  # Keep in valid pixel range
            
            # Transform and apply patch
            adv_batch_t = self.patch_transformer(
                adv_patch,
                lab_batch,
                self.cfg.model_in_sz,
                use_mul_add_gau=self.cfg.use_mul_add_gau,
                do_transforms=self.cfg.transform_patches,
                do_rotate=self.cfg.rotate_patches,
                rand_loc=self.cfg.random_patch_loc,
            )
            p_img_batch = self.patch_applier(img_batch, adv_batch_t)
            
            # Run model inference
            with torch.no_grad():
                output = self.model(p_img_batch)
            
            # Calculate loss
            loss = self.loss_fn(output, lab_batch, self.adv_patch_raw)
            ep_loss += loss.item()
            
            # Backpropagation and update
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Clamp patch values to valid range
            with torch.no_grad():
                self.adv_patch_raw.data.clamp_(0, 1)
            
            # Print progress
            if i_batch % self.cfg.print_every == 0:
                self.logger.info(f"Epoch: {epoch}, Batch: {i_batch}, Loss: {loss.item()}")
        
        # Save patch after each epoch
        self.save_patch(epoch)
        
        # Adjust learning rate
        self.scheduler.step(ep_loss)
```

## 5. Loss Function Components

The loss function is critical to the effectiveness of adversarial patches. It typically combines multiple components:

```python
def loss_fn(self, prediction, target, adv_patch):
    """Calculate the loss for patch optimization.
    
    Args:
        prediction: Model output
        target: Ground truth labels
        adv_patch: Current adversarial patch
    
    Returns:
        Combined loss value
    """
    # 1. Detection loss (main objective)
    det_loss = self.detection_loss(prediction, target)
    
    # 2. Patch appearance losses (optional regularization)
    tv_loss = self.tv_loss(adv_patch) * self.cfg.tv_loss_weight
    nps_loss = self.nps_loss(adv_patch) * self.cfg.nps_loss_weight
    
    # 3. Print component values
    if self.cfg.verbose:
        print(f"Detection loss: {det_loss.item()}")
        print(f"TV loss: {tv_loss.item()}")
        print(f"NPS loss: {nps_loss.item()}")
    
    # 4. Combine losses
    total_loss = det_loss + tv_loss + nps_loss
    
    return total_loss
```

### Detection Loss

The detection loss aims to reduce the confidence of the detector:

```python
def detection_loss(self, output, target):
    """Calculate detection loss to minimize detection confidence.
    
    Args:
        output: Model predictions (from YOLO)
        target: Ground truth labels
    
    Returns:
        Loss value that when minimized, decreases detection confidence
    """
    # Extract detection confidences from YOLO output
    if isinstance(output, list):  # YOLOv5 returns a list of predictions
        # Process each output layer
        confidences = []
        for out in output:
            # Extract objectness scores
            obj_scores = out[..., 4]
            confidences.append(obj_scores)
        
        # Combine confidences from different layers
        combined_conf = torch.cat(confidences)
        
        # Invert the loss (we want to minimize confidence)
        loss = -torch.mean(combined_conf)
    else:
        # Direct confidence extraction (older YOLO versions)
        loss = -torch.mean(output[..., 4])
    
    return loss
```

### Total Variation Loss

TV loss encourages smoothness in the patch:

```python
def tv_loss(self, patch):
    """Total variation loss to encourage patch smoothness.
    
    Args:
        patch: The adversarial patch tensor
    """
    # Calculate differences in x and y directions
    diff_x = torch.diff(patch[:, :, :], dim=2)
    diff_y = torch.diff(patch[:, :, :], dim=1)
    
    # Sum squared differences
    tv_loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
    
    return tv_loss
```

### Non-Printability Score (NPS) Loss

NPS loss ensures the patch is physically printable:

```python
def nps_loss(self, patch):
    """Non-printability score loss to ensure patch can be physically printed.
    
    Args:
        patch: The adversarial patch tensor
    """
    # Define printable RGB values
    printable_colors = torch.tensor([
        [0, 0, 0],          # Black
        [1, 1, 1],          # White
        [1, 0, 0],          # Red
        [0, 1, 0],          # Green
        [0, 0, 1],          # Blue
        # Add more printable colors as needed
    ], device=self.dev)
    
    # Reshape patch for color comparison
    patch_flat = patch.reshape(-1, 3)
    
    # Calculate distance to each printable color
    color_dists = []
    for color in printable_colors:
        # Calculate Euclidean distance
        dist = torch.sqrt(torch.sum((patch_flat - color)**2, dim=1))
        color_dists.append(dist)
    
    # Stack distances and find minimum distance to any printable color
    color_dists = torch.stack(color_dists, dim=1)
    min_dists = torch.min(color_dists, dim=1)[0]
    
    # Sum of minimum distances is the NPS loss
    nps_loss = torch.mean(min_dists)
    
    return nps_loss
```

## 6. How the Attack Works

The adversarial patch attack works through a carefully calibrated optimization process that exploits vulnerabilities in neural networks:

### 1. Feature Disruption

The patch creates patterns that:
- Generate strong activations in early layers
- Confuse feature extractors by mimicking object boundaries
- Introduce "blind spots" in attention mechanisms

```
Input Image → Feature Extractor → Detection Heads → Output
                    ↑
             Disrupted by Patch
```

### 2. Gradient-Based Optimization

The patch training process leverages the same gradient-based optimization used to train neural networks, but in reverse:

```python
# In the training loop:
loss.backward()  # Calculate gradients
optimizer.step()  # Update patch based on gradients
```

While model training minimizes loss to improve accuracy, patch optimization maximizes loss to decrease accuracy.

### 3. Physical World Considerations

For patches to work in the physical world, additional constraints are added:

```python
# Apply transformations for physical robustness
adv_batch_t = self.patch_transformer(
    adv_patch,
    lab_batch,
    self.cfg.model_in_sz,
    use_mul_add_gau=True,  # Add noise for robustness
    do_transforms=True,    # Apply geometric transforms
    do_rotate=True,        # Rotate patches
    rand_loc=True,         # Vary patch location
)
```

These transformations ensure the patch works under different:
- Viewing angles
- Lighting conditions
- Camera distances
- Partial occlusions

## 7. Visualizing the Process

During training, we can visualize the patch and its effect:

```python
def save_patch(self, epoch):
    """Save the patch and visualizations during training."""
    # Save patch image
    patch = torch.clamp(self.adv_patch_raw.detach().cpu(), 0.0, 1.0)
    patch_img = T.ToPILImage()(patch)
    patch_img.save(osp.join(self.cfg.log_dir, f"e_{epoch}.png"))
    
    # Save patched image examples
    if self.cfg.debug_mode and epoch % self.cfg.save_every == 0:
        # Apply patch to validation images for visualization
        val_results = self.evaluate_patch(epoch)
        
        # Save visualization results
        for i, (img, det) in enumerate(val_results):
            if i >= 5:  # Limit to 5 examples
                break
            img.save(osp.join(self.cfg.log_dir, "debug", f"epoch_{epoch}_example_{i}.jpg"))
```

## 8. Measuring Effectiveness

The effectiveness of the patch can be measured with several metrics:

```python
def evaluate_patch(self, epoch):
    """Evaluate patch effectiveness on validation images."""
    results = []
    
    # Process validation set
    for img_path, label_path in self.val_data:
        # Load image and apply patch
        img = Image.open(img_path).convert("RGB")
        img_tensor = T.ToTensor()(img).unsqueeze(0).to(self.dev)
        
        # Load labels
        with open(label_path, 'r') as f:
            labels = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts])
        
        label_tensor = torch.tensor(labels, device=self.dev).float().unsqueeze(0)
        
        # Apply patch
        adv_patch = torch.clamp(self.adv_patch_raw.detach(), 0.0, 1.0)
        adv_batch_t = self.patch_transformer(adv_patch, label_tensor, img.size)
        patched_img = self.patch_applier(img_tensor, adv_batch_t)
        
        # Run detection
        with torch.no_grad():
            detections = self.model(patched_img)
        
        # Process detections
        # Original vs. Patched detection metrics
        # ...
        
        # Convert to PIL for visualization
        patched_img_pil = T.ToPILImage()(patched_img.squeeze(0).cpu())
        
        # Draw detections on image
        drawn_img = self.draw_detections(patched_img_pil, detections)
        
        results.append((drawn_img, detections))
    
    # Calculate overall metrics
    # Detection rate, confidence reduction, etc.
    
    return results
```

## 9. Practical Applications and Defense

Understanding adversarial patches helps develop more robust models:

1. **Adversarial Training**: Train models on patched images to build resistance
2. **Detection Systems**: Implement patch detection algorithms
3. **Physical Defenses**: Careful camera placement and scene design to minimize attack surfaces

The code for adversarial training:

```python
# From train_adversarial.py
def main(args):
    """Main function for adversarial training."""
    # Step 1: Create patched dataset
    if not args.skip_patching:
        print("Creating adversarially patched dataset...")
        apply_patch_to_dataset(
            patch_path=args.patch,
            dataset_dir=args.dataset,
            output_dir=args.patched_dataset,
            patch_size=args.patch_size,
            target_size_frac=args.target_size_frac,
            patch_alpha=args.patch_alpha,
            image_size=args.image_size,
            device=args.device,
            filter_class_ids=args.filter_classes,
            apply_to_splits=args.splits
        )
    
    # Step 2: Train YOLOv5 on patched dataset
    if not args.skip_training:
        print("Training YOLOv5 on patched dataset...")
        # Set up training command
        train_cmd = [
            "python", "train.py",
            "--weights", args.weights,
            "--data", f"{args.patched_dataset}/data.yaml",
            "--hyp", args.hyp,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--img", str(args.image_size[0]),
            "--project", args.project,
            "--name", args.name,
            "--workers", str(args.workers),
            "--device", args.device.split(':')[1] if ':' in args.device else args.device
        ]
        
        # Run training process
        subprocess.run(train_cmd)
```

## 10. Example Training Configuration

A typical configuration for patch training:

```python
# Example configuration from configs/patch/example.yaml
model_config:
  model_def: "models/yolov5s.yaml"
  weightfile: "weights/yolov5s.pt"
  model_in_sz: [640, 640]
  class_list: ["car", "bus", "truck"]
  conf_thresh: 0.4
  nms_thresh: 0.45

patch_config:
  init_mode: "random"  # random, gray, or file
  init_size: [300, 300]
  patch_path: "logs/patch_attack/e_0.png"  # Used if init_mode is "file"
  transform_patches: true
  rotate_patches: true
  target_size_frac: [0.2, 0.4]  # Patch size relative to bounding box
  random_patch_loc: true  # Random positioning around target
  patch_alpha: 1.0  # Transparency (1.0 = opaque)

training:
  start_epoch: 0
  epochs: 100
  batch_size: 8
  start_learning_rate: 0.03
  scheduler_patience: 10
  scheduler_factor: 0.5
  
  # Loss weights
  tv_loss_weight: 2.5
  nps_loss_weight: 0.1
  
  # Additional settings
  use_mul_add_gau: true
  print_every: 50
  save_every: 20
  log_dir: "logs/patch_attack"
  debug_mode: true
```

## Patch Position During Training

The position of the patch during training is determined by the target object's position from the labels, with some randomization for robustness. Here's how it works:

1. **Base Position from Labels**:
```python
# Get target position from labels
target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # x center
target_y = lab_batch[:, :, 2].view(np.prod(batch_size))  # y center
```
The base position comes from the YOLO format labels, where:
- `lab_batch[:, :, 1]` is the x-center (normalized 0-1)
- `lab_batch[:, :, 2]` is the y-center (normalized 0-1)

2. **Random Offset for Robustness**:
```python
if rand_loc:
    # Add random offset based on object size
    off_x = w_size.view(np.prod(batch_size)) * torch.empty(w_size.view(np.prod(batch_size)).size(), device=self.dev).uniform_(*self.x_off_loc)
    target_x = target_x + off_x
    off_y = h_size.view(np.prod(batch_size)) * torch.empty(h_size.view(np.prod(batch_size)).size(), device=self.dev).uniform_(*self.y_off_loc)
    target_y = target_y + off_y
```
The patch position is slightly randomized around the target object:
- `x_off_loc=[-0.25, 0.25]`: Random offset of ±25% of object width
- `y_off_loc=[-0.25, 0.25]`: Random offset of ±25% of object height

3. **Position During Training Loop**:
```python
# In train_patch.py
for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
    for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
        # Get current patch
        adv_patch = self.adv_patch_raw.clone()
        
        # Transform and position patch based on current batch labels
        adv_batch_t = self.patch_transformer(
            adv_patch,
            lab_batch,  # Labels determine patch positions
            self.cfg.model_in_sz,
            rand_loc=True  # Enable random positioning
        )
        
        # Apply patch to images
        p_img_batch = self.patch_applier(img_batch, adv_batch_t)
        
        # Calculate loss and update patch content
        loss = self.loss_fn(output, lab_batch, self.adv_patch_raw)
        loss.backward()
        self.optimizer.step()
```

Important points about patch positioning:

1. **Position vs Content**:
   - The patch position is determined by the labels and randomization
   - What gets optimized during training is the patch's content (pixel values)
   - The position is not a trainable parameter

2. **Why This Approach?**:
   - The patch needs to work regardless of where it's placed on the target object
   - Random positioning during training helps create a more robust patch
   - The patch learns to be effective across different positions

3. **Position Constraints**:
```python
# Ensure patch stays within image boundaries
x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(img_size[1], x2)
y2 = min(img_size[0], y2)
```
The patch is always constrained to stay within the image boundaries.

4. **Training Process**:
   - For each batch:
     1. Get new images and their labels
     2. Calculate patch positions based on labels
     3. Apply random offsets
     4. Transform and apply patch
     5. Calculate loss
     6. Update patch content (not position)
   - This process repeats for many epochs

5. **Visualization of Position Updates**:
```python
def save_patch(self, epoch):
    """Save visualization of patch positions during training."""
    if self.cfg.debug_mode and epoch % self.cfg.save_every == 0:
        # Apply patch to validation images
        val_results = self.evaluate_patch(epoch)
        
        # Save images showing patch positions
        for i, (img, det) in enumerate(val_results):
            if i >= 5:  # Limit to 5 examples
                break
            # Draw bounding boxes and patch positions
            img_with_boxes = self.draw_detections(img, det)
            img_with_boxes.save(osp.join(self.cfg.log_dir, "debug", 
                                       f"epoch_{epoch}_example_{i}.jpg"))
```

This approach ensures that:
1. The patch is always placed on or near the target object
2. The patch learns to be effective across different positions
3. The training process focuses on optimizing the patch content rather than its position
4. The resulting patch is robust to slight variations in placement

## Conclusion

Adversarial patches represent a powerful attack vector against object detection systems. By understanding how they work and the training process behind them, we can develop more robust models and appropriate countermeasures.

The process of y' = patch + (x, y, w, h) might seem simple in concept, but the optimization involves sophisticated techniques to ensure the patch works consistently across various conditions while remaining physically realizable. 