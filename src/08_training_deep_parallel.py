import os, sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from sklearn.model_selection import train_test_split
from custom_data_generator import imageLoader
from models import AR2B_UNet_DeepSup
from write_options import DualStream
from losses import dice_loss, dice_coef, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing, accuracy, sensitivity, specificity, dice_score, precision
from losses import CombinedLoss

# At the start of your code (before any modifications to sys.stdout and sys.stderr):
original_stdout = sys.stdout
original_stderr = sys.stderr

# Open the output file to save training metrics
output_file = open("08_output.txt", "w")
error_file = open("08_errors.txt", "w")

# Redirect Output to files and screen
sys.stdout = DualStream(sys.stdout, output_file)
sys.stderr = DualStream(sys.stderr, error_file)


# Define the directories for the images and masks
"""WHEN USING GPU ON REMOTE SSH"""
TRAIN_IMG_DIR = "../data/training/images"
TRAIN_MASK_DIR = "../data/training/masks"

VAL_IMG_DIR = "../data/test/images"
VAL_MASK_DIR = "../data/test/masks"

"""WHEN USING CPU ON LOCAL"""
#TRAIN_IMG_DIR = "./data/images"  # "../data/nnunet_raw_data_base/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans_3d_fullres"
#TRAIN_MASK_DIR = "./data/masks"  # "../data/nnunet_raw_data_base/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans_3d_fullres"

# Get list of image and mask names
train_img_list = sorted(os.listdir(TRAIN_IMG_DIR))
train_mask_list = sorted(os.listdir(TRAIN_MASK_DIR))

# Remove files not ending with ".npy"
train_img_list = [file for file in train_img_list if file.endswith('.npy')]
train_mask_list = [file for file in train_mask_list if file.endswith('.npy')]

# Check Count
if(len(train_img_list) != len(train_mask_list)):
    print("Images and Masks not equal in size")
    input()

# Split the data into training and validation sets
# train_img_list, val_img_list, train_mask_list, val_mask_list = train_test_split(
#     train_img_list, train_mask_list, test_size=0.2, random_state=42)

# Get list of image and mask names
val_img_list = sorted(os.listdir(VAL_IMG_DIR))
val_mask_list = sorted(os.listdir(VAL_MASK_DIR))

# Remove files not ending with ".npy"
val_img_list = [file for file in val_img_list if file.endswith('.npy')]
val_mask_list = [file for file in val_mask_list if file.endswith('.npy')]

# Check Count
if(len(val_img_list) != len(val_mask_list)):
    print("Validation Images and Masks not equal in size")
    input()


# Print the sizes of the training and validation sets
print(f"Number of training images: {len(train_img_list)}")
print(f"Number of validation images: {len(val_img_list)}")
"""
## Find distribution/weights of each class
columns = ['0', '1', '2', '3']
df = []

for img in range(len(train_mask_list)):
    temp_mask = np.load(TRAIN_MASK_DIR + "/" + train_mask_list[img])
    temp_mask = np.argmax(temp_mask, axis=0)
    val, counts = np.unique(temp_mask, return_counts=True)
    conts_dict = dict(zip(columns, counts))

    df.append(conts_dict)


df = pd.DataFrame(df)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

# Class weights calculation: _samples / (_classes * _samples_for_class)
wt0 = round(total_labels / (n_classes * label_0), 2)
wt1 = round(total_labels / (n_classes * label_1), 2)
wt2 = round(total_labels / (n_classes * label_2), 2)
wt3 = round(total_labels / (n_classes * label_3), 2)

print(wt0, wt1, wt2, wt3)
# 0.26 30.29 8.73 33.9

# Compute class weights here (code omitted)
class_weights = {0: wt0, 1: wt1, 2: wt2, 3: wt3}
"""
class_weights = {0: 0.26, 1: 33.18, 2: 8.57, 3: 23.45}

# Set Learning rate
LR = 0.0001

# Define Batch Size
BATCH_SIZE = 4
accumulation_steps = 30

# Specify GPU ids to use
gpus = [1, 2, 4] #[3, 5, 6, 7] 
# Set device to be cuda for the specified GPUs
device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')

# Check if CUDA is available
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define the model and move it to the GPU
model = AR2B_UNet_DeepSup(in_channels=4, num_classes=4)

# Move model to GPU
model = model.to(device)

# Use DataParallel to spread model across multiple GPUs
if len(gpus) > 1:
  print(f"Using GPUs {gpus}")
  model = nn.DataParallel(model, device_ids=gpus)

print("model created")

num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Convert the class weights to a PyTorch tensor and move it to the device (CPU or GPU)
#weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], device=device, dtype=torch.float)

# Define the CrossEntropyLoss with class weights
#criterion = nn.CrossEntropyLoss(weight=weights).to(device)
"""OR MAYBE USE THE DICE LOSS"""

class_weights_list = [class_weights[i] for i in range(len(class_weights))]
criterion = CombinedLoss(num_classes=3, class_weights=class_weights_list, device_num = gpus[0]).to(device)

# Define the DataLoaders
train_data_gen = imageLoader(TRAIN_IMG_DIR, train_img_list, TRAIN_MASK_DIR, train_mask_list, BATCH_SIZE, num_workers = 8)
val_data_gen = imageLoader(VAL_IMG_DIR, val_img_list, VAL_MASK_DIR, val_mask_list, BATCH_SIZE, num_workers=8)  # Define the validation DataLoader here


# Initialize the gradient scaler for amp
scaler = amp.GradScaler()


print("Model Training Step") 

# Check if a saved model exists and load it
model_path = '../saved_models/08_best_model_deep.pt'
if os.path.isfile(model_path):
    saved_model = torch.load(model_path)
    # If the saved model was wrapped with nn.DataParallel, load it as-is
    if 'module.' in list(saved_model.keys())[0]:
        model.load_state_dict(saved_model)
    # If the saved model was not wrapped with nn.DataParallel, load it into the unwrapped model
    else:
        model.module.load_state_dict(saved_model)
    print("Loaded model from disk")

# Initialize results DataFrame
results_df = pd.DataFrame(columns=[
    'Epoch', 'Loss', 'Accuracy', 
    'Dice Coef (0)', 'Dice Coef (1)', 'Dice Coef (2)', 'Dice Coef (3)',
    'Dice Coef Necrotic', 'Dice Coef Edema', 'Dice Coef Enhancing',
    'Sensitivity (0)', 'Sensitivity (1)', 'Sensitivity (2)', 'Sensitivity (3)',
    'Specificity (0)', 'Specificity (1)', 'Specificity (2)', 'Specificity (3)',
    'Precision (0)', 'Precision (1)', 'Precision (2)', 'Precision (3)'
])

# Training loop
best_loss = float('inf')
num_classes = 4
for epoch in range(1000):
    model.train()
    epoch_loss = 0

    dice_score_necrotic = 0
    dice_score_edema = 0
    dice_score_enhancing = 0

    # Metrics initialization
    total_accuracy = 0
    total_sensitivity = [0] * num_classes
    total_specificity = [0] * num_classes
    total_precision = [0] * num_classes
    total_dice_score = [0] * num_classes


    for batch_num, (imgs, masks) in enumerate(train_data_gen):
        imgs, masks = imgs.to(device), masks.to(device)

        # Forward pass
        optimizer.zero_grad()

        with amp.autocast():
            main_logit, aux_logit_1, aux_logit_2, aux_logit_3, aux_logit_4 = model(imgs) # Probability as Model Outputs - REQUIRED FOR CROSS ENTROPY LOSS
            masks = torch.argmax(masks, dim=1)  # To remove one-hot encoding

            # Compute loss for each of the outputs
            main_loss = criterion(main_logit, masks)
            aux_loss_1 = criterion(aux_logit_1, masks)
            aux_loss_2 = criterion(aux_logit_2, masks)
            aux_loss_3 = criterion(aux_logit_3, masks)
            aux_loss_4 = criterion(aux_logit_4, masks)

            # Average the losses
            loss = main_loss + aux_loss_1 + aux_loss_2 + aux_loss_3 + aux_loss_4
            loss /= 5


            outputs = torch.argmax(main_logit, dim=1) # To remove one-hot encoding

            
            # Compute the mean loss for optimization
            loss = loss.mean()
            #print("Mean Loss:", loss)


        if np.isnan(loss.item()):
            sys.stderr.write(f"NaN loss found in batch {batch_num}\n")
            
            # Print image and mask filenames
            img_names = train_data_gen.dataset.img_list[batch_num*BATCH_SIZE : (batch_num+1)*BATCH_SIZE] 
            mask_names = train_data_gen.dataset.mask_list[batch_num*BATCH_SIZE : (batch_num+1)*BATCH_SIZE]
            
            sys.stderr.write("Image names:\n")
            sys.stderr.write(str(img_names))
            sys.stderr.write("\n")
            sys.stderr.write("Mask names:\n")
            sys.stderr.write(str(mask_names))
            sys.stderr.write("\n")

        # Backward pass and optimization
        scaler.scale(loss).backward(retain_graph=True)



        # Perform optimization after accumulation_steps
        if (batch_num + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        print("Batch:", batch_num,", Combined Loss:", loss,", Dice Loss:", dice_loss(masks, outputs).mean().item())

        epoch_loss += loss.item()

        # Calculate metrics for each class
        total_accuracy += accuracy(outputs, masks).item()
        for class_id in range(num_classes):
            total_sensitivity[class_id] += sensitivity(outputs, masks, class_id).item()
            total_specificity[class_id] += specificity(outputs, masks, class_id).item()
            total_precision[class_id] += precision(outputs, masks, class_id).item()
            total_dice_score[class_id] += dice_score(outputs, masks, class_id).item()

        dice_score_necrotic += dice_coef_necrotic(masks, outputs).item()
        dice_score_edema += dice_coef_edema(masks, outputs).item()
        dice_score_enhancing += dice_coef_enhancing(masks, outputs).item()

    # Divide by the number of classes to get the average
    epoch_loss /= len(train_data_gen)
    total_accuracy /= len(train_data_gen)
    total_sensitivity = [sens / len(train_data_gen) for sens in total_sensitivity]
    total_specificity = [spec / len(train_data_gen) for spec in total_specificity]
    total_precision = [prec / len(train_data_gen) for prec in total_precision]
    total_dice_score = [dice / len(train_data_gen) for dice in total_dice_score]
    dice_score_necrotic /= len(train_data_gen)
    dice_score_edema /= len(train_data_gen)
    dice_score_enhancing /= len(train_data_gen)
    print(f"----------Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {total_accuracy}, Dice Coef: {total_dice_score}, "
        f"Dice Coef Necrotic: {dice_score_necrotic}, Dice Coef Edema: {dice_score_edema}, "
        f"Dice Coef Enhancing: {dice_score_enhancing}, Sensitivity: {total_sensitivity}, "
        f"Specificity: {total_specificity}, "
        f"Precision: {total_precision}")

    # Append results to DataFrame
    new_row = {'Epoch': epoch + 1, 
            'Loss': epoch_loss, 
            'Accuracy': total_accuracy,
            'Dice Coef (0)': total_dice_score[0],
            'Dice Coef (1)': total_dice_score[1],  
            'Dice Coef (2)': total_dice_score[2],
            'Dice Coef (3)': total_dice_score[3],
            'Dice Coef Necrotic': dice_score_necrotic,
            'Dice Coef Edema': dice_score_edema,
            'Dice Coef Enhancing': dice_score_enhancing,
            'Sensitivity (0)': total_sensitivity[0],
            'Sensitivity (1)': total_sensitivity[1],
            'Sensitivity (2)': total_sensitivity[2], 
            'Sensitivity (3)': total_sensitivity[3],
            'Specificity (0)': total_specificity[0],
            'Specificity (1)': total_specificity[1],
            'Specificity (2)': total_specificity[2],
            'Specificity (3)': total_specificity[3],
            'Precision (0)': total_precision[0],
            'Precision (1)': total_precision[1],
            'Precision (2)': total_precision[2],
            'Precision (3)': total_precision[3]}
    
    results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    #print(new_row)
    

    # Validation step after every 50 epochs
    if (epoch + 1) % 50 == 0:
        model.eval()  # Set the model to evaluation mode

        # Initialize metrics for validation
        val_loss = 0
        val_total_accuracy = 0
        val_total_sensitivity = [0] * num_classes
        val_total_specificity = [0] * num_classes
        val_total_precision = [0] * num_classes
        val_total_dice_score = [0] * num_classes

        val_dice_score_necrotic = 0
        val_dice_score_edema = 0
        val_dice_score_enhancing = 0
        # ... other metrics initialization ...

        with torch.no_grad():  # Disable gradients for validation
            for val_batch_num, (val_imgs, val_masks) in enumerate(val_data_gen):
                val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)

                val_main_logit, val_aux_logit_1, val_aux_logit_2, val_aux_logit_3, val_aux_logit_4 = model(val_imgs) # Probability as Model Outputs - REQUIRED FOR CROSS ENTROPY LOSS
                val_masks = torch.argmax(val_masks, dim=1)  # To remove one-hot encoding

                # Compute loss for each of the outputs
                val_main_loss = criterion(val_main_logit, val_masks)
                val_aux_loss_1 = criterion(val_aux_logit_1, val_masks)
                val_aux_loss_2 = criterion(val_aux_logit_2, val_masks)
                val_aux_loss_3 = criterion(val_aux_logit_3, val_masks)
                val_aux_loss_4 = criterion(val_aux_logit_4, val_masks)

                # Average the losses
                val_loss_batch = val_main_loss + val_aux_loss_1 + val_aux_loss_2 + val_aux_loss_3 + val_aux_loss_4
                val_loss_batch /= 5


                val_outputs = torch.argmax(val_main_logit, dim=1) # To remove one-hot encoding

                
                # Compute the mean loss for optimization
                val_loss += val_loss_batch.mean().item()


                # if np.isnan(val_loss):
                #     sys.stderr.write(f"NaN loss found in Validation batch {val_batch_num}\n")  

                #     # Print image and mask filenames
                #     img_names = val_data_gen.dataset.img_list[val_batch_num*BATCH_SIZE : (val_batch_num+1)*BATCH_SIZE]
                #     mask_names = val_data_gen.dataset.mask_list[val_batch_num*BATCH_SIZE : (val_batch_num+1)*BATCH_SIZE]

                #     sys.stderr.write("Image names:\n")
                #     sys.stderr.write(str(img_names))
                #     sys.stderr.write("\n")
                #     sys.stderr.write("Mask names:\n")
                #     sys.stderr.write(str(mask_names))
                #     sys.stderr.write("\n")

                # Calculate metrics for validation
                val_total_accuracy += accuracy(val_outputs, val_masks).item()
                for class_id in range(num_classes):
                    val_total_sensitivity[class_id] += sensitivity(val_outputs, val_masks, class_id).item()
                    val_total_specificity[class_id] += specificity(val_outputs, val_masks, class_id).item()
                    val_total_precision[class_id] += precision(val_outputs, val_masks, class_id).item()
                    val_total_dice_score[class_id] += dice_score(val_outputs, val_masks, class_id).item()

                val_dice_score_necrotic += dice_coef_necrotic(val_masks, val_outputs).item()
                val_dice_score_edema += dice_coef_edema(val_masks, val_outputs).item()
                val_dice_score_enhancing += dice_coef_enhancing(val_masks, val_outputs).item()

                    # ... other per-class metrics ...

        # Calculate average validation metrics
        val_loss /= len(val_data_gen)
        val_total_accuracy /= len(val_data_gen)
        val_total_sensitivity = [sens / len(val_data_gen) for sens in val_total_sensitivity]
        val_total_specificity = [spec / len(val_data_gen) for spec in val_total_specificity]
        val_total_precision = [sens / len(val_data_gen) for sens in val_total_precision]
        val_total_dice_score = [spec / len(val_data_gen) for spec in val_total_dice_score]
        val_dice_score_necrotic /= len(train_data_gen)
        val_dice_score_edema /= len(train_data_gen)
        val_dice_score_enhancing /= len(train_data_gen)
        # ... other metrics ...

        # Print validation results
        print(f"-----------Validation Epoch {epoch+1}, Loss: {val_loss}, Accuracy: {val_total_accuracy}, Dice Coef: {val_total_dice_score}, "
            f"Dice Coef Necrotic: {val_dice_score_necrotic}, Dice Coef Edema: {val_dice_score_edema}, "
            f"Dice Coef Enhancing: {val_dice_score_enhancing}, Sensitivity: {val_total_sensitivity}, "
            f"Specificity: {val_total_specificity}, "
            f"Precision: {val_total_precision}")
            # ... other metrics print statements ...

        # Append results to DataFrame
        new_row = {'Epoch': "Validation "+ str(epoch + 1), 
                'Loss': val_loss, 
                'Accuracy': val_total_accuracy,
                'Dice Coef (0)': val_total_dice_score[0],
                'Dice Coef (1)': val_total_dice_score[1],  
                'Dice Coef (2)': val_total_dice_score[2],
                'Dice Coef (3)': val_total_dice_score[3],
                'Dice Coef Necrotic': val_dice_score_necrotic,
                'Dice Coef Edema': val_dice_score_edema,
                'Dice Coef Enhancing': val_dice_score_enhancing,
                'Sensitivity (0)': val_total_sensitivity[0],
                'Sensitivity (1)': val_total_sensitivity[1],
                'Sensitivity (2)': val_total_sensitivity[2], 
                'Sensitivity (3)': val_total_sensitivity[3],
                'Specificity (0)': val_total_specificity[0],
                'Specificity (1)': val_total_specificity[1],
                'Specificity (2)': val_total_specificity[2],
                'Specificity (3)': val_total_specificity[3],
                'Precision (0)': val_total_precision[0],
                'Precision (1)': val_total_precision[1],
                'Precision (2)': val_total_precision[2],
                'Precision (3)': val_total_precision[3]}
        
        results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        #print(new_row)
        model.train()  # Switch back to training mode

    # Save the DataFrame to a CSV file after every epoch
    results_df.to_csv('08_training_results.csv', index=False)

    # Save the best model with lowest loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        if isinstance(model, nn.DataParallel):
            # Save the state dictionary of the underlying model if it's wrapped with nn.DataParallel
            print("Saving with nn.DataParallel")
            torch.save(model.module.state_dict(), '../saved_models/08_best_model_deep.pt')
        else:
            # Save the state dictionary as-is if the model is not wrapped with nn.DataParallel
            print("Saving without nn.DataParallel")
            torch.save(model.state_dict(), '../saved_models/08_best_model_deep.pt')

# Save the final model
if isinstance(model, nn.DataParallel):
    print("Saving final model with nn.DataParallel")
    torch.save(model.module.state_dict(), '../saved_models/08_final_model_deep.pt')
else:
    print("Saving final model without nn.DataParallel")
    torch.save(model.state_dict(), '../saved_models/08_final_model_deep.pt')


# Close both output files
output_file.close()
error_file.close()

# At the end of your code, revert back to original stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr

