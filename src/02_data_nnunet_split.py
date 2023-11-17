import os, random, glob
import nibabel as nib

# Set random seed for reproducibility
random.seed(77) 

TRAINING_DATASET_PATH = "../input/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
OUT_TRAIN_NNUNET_IMAGES = '../data/training/nnUNet_raw/Dataset001_BRATS/imagesTr/'
OUT_TRAIN_NNUNET_LABELS = '../data/training/nnUNet_raw/Dataset001_BRATS/labelsTr/'
OUT_TEST_NNUNET_IMAGES = '../data/test/nnUNet_raw/imagesTs/' 
OUT_TEST_NNUNET_LABELS = '../data/test/nnUNet_raw/labelsTs/'

# Get list of subfolders
def list_subfolders(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

total_cases = list_subfolders(TRAINING_DATASET_PATH)

# Split into train/val
num_cases = len(total_cases)
num_train = int(0.85 * num_cases)
train_cases = random.sample(total_cases, num_train) 
test_cases = [case for case in total_cases if case not in train_cases]

print([len(i) for i in [total_cases, train_cases, test_cases]])

def process_case(case, case_num, out_img_dir, out_label_dir):

    case_path = os.path.join(TRAINING_DATASET_PATH, case)
    
    # Load and save images
    imgs = {'t1n': '*t1n.nii.gz', 't1c': '*t1c.nii.gz', 
            't2w': '*t2w.nii.gz', 'flair': '*t2f.nii.gz'}
    
    for i, (code, fname) in enumerate(imgs.items()):
        print(case_path, fname)
        img = nib.load(glob.glob(case_path + '/' + fname)[0])
        nib.save(img, os.path.join(out_img_dir, f'BRATS_{case_num:04d}_000{i}.nii.gz'))
        print(f"Saved {fname} image as BRATS_{case_num:04d}_000{i}.nii.gz")

    # Load and save label
    seg = nib.load(glob.glob(case_path + '/' + '*seg.nii.gz')[0])
    nib.save(seg, os.path.join(out_label_dir, f'BRATS_{case_num:04d}.nii.gz'))
    print(f"Saved *seg.nii.gz mask as BRATS_{case_num:04d}.nii.gz")
        
print("TRAINING CASES")
# Process training cases
for i, case in enumerate(train_cases):
    process_case(case, i, OUT_TRAIN_NNUNET_IMAGES, OUT_TRAIN_NNUNET_LABELS)
    print(f"Saved {i} of {num_train} training cases")
    
print("TEST CASES")
# Process test cases
for i, case in enumerate(test_cases):
    process_case(case, i + num_train, OUT_TEST_NNUNET_IMAGES, OUT_TEST_NNUNET_LABELS)
    print(f"Saved {i+num_train} as test cases")

print("COMPLETE")
