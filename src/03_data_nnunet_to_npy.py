import glob
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

"""Directories from nnUNET raw folder"""
IN_TRAIN_NNUNET_IMAGES = '../data/training/nnUNet_raw/Dataset001_BRATS/imagesTr/'
IN_TRAIN_NNUNET_LABELS = '../data/training/nnUNet_raw/Dataset001_BRATS/labelsTr/'
IN_TEST_NNUNET_IMAGES = '../data/test/nnUNet_raw/Dataset001_BRATS/imagesTs/' 
IN_TEST_NNUNET_LABELS = '../data/test/nnUNet_raw/Dataset001_BRATS/labelsTs/'

"""Directories from normal train/test folder"""
OUT_TRAIN_IMAGES = '../data/training/images'
OUT_TRAIN_LABELS = '../data/training/masks'
OUT_TEST_IMAGES = '../data/test/images' 
OUT_TEST_LABELS = '../data/test/masks'

# Check if CUDA (GPU) is available and use it, otherwise, fall back to CPU
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"Using",device)

# Function to scale an image using MinMaxScaler
def scale_image(image):
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

# Function to convert the .nii.gz files to .npy combined
# Operation - Scaling, Cropping, Combining, Permutation (for PyTorch Models)
def nnunet_to_npy(IN_IMAGES, IN_LABELS, OUT_IMAGES, OUT_LABELS):

    t1n_list = sorted(glob.glob(IN_IMAGES+'/*0000.nii.gz'))
    t1c_list = sorted(glob.glob(IN_IMAGES+'/*0001.nii.gz'))
    t2w_list = sorted(glob.glob(IN_IMAGES+'/*0002.nii.gz'))
    t2f_list = sorted(glob.glob(IN_IMAGES+'/*0003.nii.gz'))
    mask_list = sorted(glob.glob(IN_LABELS+'/*.nii.gz'))

    print("Number of t1n files:", len(t1n_list))
    print("Number of t1c files:", len(t1c_list))
    print("Number of t2w files:", len(t2w_list))
    print("Number of t2f files:", len(t2f_list))
    print("Number of mask files:", len(mask_list))

    if(not ( len(t1n_list) == len(t1c_list) == len(t2w_list) == len(t2f_list) )):
        print("Length not equal! Quitting!")
        return

    for img in range(len(t2f_list)):
        # T1n image
        temp_image_t1n = nib.load(t1n_list[img]).get_fdata()
        temp_image_t1n = scale_image(temp_image_t1n)

        # T1c image
        temp_image_t1c = nib.load(t1c_list[img]).get_fdata()
        temp_image_t1c = scale_image(temp_image_t1c)

        # T2w image
        temp_image_t2w = nib.load(t2w_list[img]).get_fdata()
        temp_image_t2w = scale_image(temp_image_t2w)

        # T2f image
        temp_image_t2f = nib.load(t2f_list[img]).get_fdata()
        temp_image_t2f = scale_image(temp_image_t2f)

        # Mask Image
        temp_mask = nib.load(mask_list[img]).get_fdata()
        temp_mask = temp_mask.astype(np.float64)

        # Cropping the images and the mask
        temp_image_t1n = temp_image_t1n[56:184, 56:184, 13:141]
        temp_image_t1c = temp_image_t1c[56:184, 56:184, 13:141]
        temp_image_t2w = temp_image_t2w[56:184, 56:184, 13:141]
        temp_image_t2f = temp_image_t2f[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]

        # Convert NumPy arrays to PyTorch tensors and move them to the GPU if available
        temp_image_t1n = torch.tensor(temp_image_t1n, dtype=torch.float32, device=device)
        temp_image_t1c = torch.tensor(temp_image_t1c, dtype=torch.float32, device=device)
        temp_image_t2w = torch.tensor(temp_image_t2w, dtype=torch.float32, device=device)
        temp_image_t2f = torch.tensor(temp_image_t2f, dtype=torch.float32, device=device)
        temp_mask = torch.tensor(temp_mask, dtype=torch.long, device=device)

        # Stack the images into a single volume
        temp_combined_images = torch.stack([temp_image_t1n, temp_image_t1c, temp_image_t2w, temp_image_t2f], dim=0)

        # Check if at least 1% of the volume has been annotated
        if (1 - (torch.sum(temp_mask == 0) / torch.prod(torch.tensor(temp_mask.shape)))) > 0.01:
            print(f"Saving {img} of {len(t2f_list)}")

            # Convert the temp_mask to one-hot using PyTorch
            temp_mask = F.one_hot(temp_mask, num_classes=4).permute(3, 0, 1, 2)
            
            # Save the images and the mask
            np.save(OUT_IMAGES+"/image_" + str(img) + ".npy", temp_combined_images.cpu().numpy())
            np.save(OUT_LABELS+"/mask_" + str(img) + ".npy", temp_mask.cpu().numpy().astype(np.uint8))
                        
            # Free up GPU memory after processing
            torch.cuda.empty_cache()

        else:
            print("Not enough labeled data")
            

print("PREPROCESS TRAINING CASES")
nnunet_to_npy(IN_TRAIN_NNUNET_IMAGES, IN_TRAIN_NNUNET_LABELS, OUT_TRAIN_IMAGES, OUT_TRAIN_LABELS)

print("PREPROCESS TEST CASES")
nnunet_to_npy(IN_TEST_NNUNET_IMAGES, IN_TEST_NNUNET_LABELS, OUT_TEST_IMAGES, OUT_TEST_LABELS)




