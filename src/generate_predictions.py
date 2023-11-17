import os
import sys
import torch
import numpy as np
import pandas as pd

from models import AR2B_UNet, AR2B_UNet_DeepSup, Swin_AR2B_DeepSup_UNet
from custom_data_generator import imageLoader

# Dataset paths
VAL_IMG_DIR = "../data/test/images/"
VAL_MASK_DIR = "../data/test/masks/" 

# Get validation image and mask lists
val_img_list = sorted(os.listdir(VAL_IMG_DIR))
val_mask_list = sorted(os.listdir(VAL_MASK_DIR))

# Remove non-npy files
val_img_list = [f for f in val_img_list if f.endswith(".npy")]
val_mask_list = [f for f in val_mask_list if f.endswith(".npy")]

# Define model paths
MODEL_PATHS = {
    "Baseline": "../saved_models/07_best_model_base.pt",
    "DeepSup": "../saved_models/08_best_model_deep_backup.pt",
    "Swin_DeepSup": "../saved_models/09_best_model_swin.pt"
}

# Instantiate models
models = {}
for name, model_path in MODEL_PATHS.items():
    if name == "Baseline":
        model = AR2B_UNet(in_channels=4, num_classes=4)
    elif name == "DeepSup":
        model = AR2B_UNet_DeepSup(in_channels=4, num_classes=4)
    elif name == "Swin_DeepSup":
        model = Swin_AR2B_DeepSup_UNet(in_channels=4, num_classes=4)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    models[name] = model

# Create DataLoader
val_dataloader = imageLoader(VAL_IMG_DIR, val_img_list[:15], VAL_MASK_DIR, val_mask_list[:15], batch_size=1)

# Set GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Evaluation loop
with torch.no_grad():
  for batch_num, (val_imgs, val_masks) in enumerate(val_dataloader):
    val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)

    # Save first images, gt and predictions
    print(f"Saving {batch_num}")
    np.save(f"../predictions/val_img_{batch_num}.npy", val_imgs.cpu().numpy())
    np.save(f"../predictions/val_gt_{batch_num}.npy", val_masks.cpu().numpy())
    for name in models:
        model = models[name].to(device) # Move to GPU
        model_output, *_ = model(val_imgs) # Get the output
        np.save(f"../predictions/val_pred_{name}_{batch_num}.npy",  model_output.cpu().numpy())
    print(f"Saved {batch_num}")
        
