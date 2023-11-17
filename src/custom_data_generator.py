import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BratsDataset(Dataset):
    """
    PyTorch dataset class to load BraTS images and masks.
    """
    
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        """
        Initialize the dataset with the following:
        img_dir (str): Path to the image directory
        img_list (list): List of image filenames
        mask_dir (str): Path to the mask directory 
        mask_list (list): List of mask filenames
        """
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.img_list)
    
    def __getitem__(self, idx):
        """
        Return the image and corresponding mask for the given index
        """
        img_name = self.img_list[idx]
        mask_name = self.mask_list[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Check if image file exists and is not empty
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise ValueError(f"Image file {img_path} does not exist or is empty")
            
        # Check if mask file exists and is not empty
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            raise ValueError(f"Mask file {mask_path} does not exist or is empty")
            
        try:
            img = np.load(img_path)
        except Exception as e:
            raise ValueError(f"Error loading image file {img_path}: {e}")
        
        try:
            mask = np.load(mask_path) 
        except Exception as e:
            raise ValueError(f"Error loading mask file {mask_path}: {e}")
        
        return img, mask
        
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, num_workers=24):
    """
    Function to load the BraTS dataset using DataLoader
    """
    dataset = BratsDataset(img_dir, img_list, mask_dir, mask_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, 
                            pin_memory=True)
                            
    return dataloader
