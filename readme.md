# Brain Tumor Segmentation Project README

This repository contains the code implementation of my Master's dissertation project at Brunel University London. The project focuses on developing advanced methods for automated segmentation of brain tumors from multimodal MRI scans by integrating modern mechanisms like Attention, Multi-Objective Losses and Swin Transformers into the traditional convolutional neural networks (CNNs) based U-Net architecture. Key challenges addressed include localizing indistinct tumor boundaries, handling class imbalances in medical datasets, and efficiently capturing global context. This work integrates attention mechanisms, multi-scale deep supervision, multi-objective loss functions, and Swin-Transformers into CNN-based architectures.

## Key Features:

**Dataset:** The Brain Tumor Segmentation (BraTS) 2023 dataset with T1, T1-contrasted, T2, and FLAIR MRI scans, including expert annotations for 1251 cases.

**Model Architectures:**
1. **AR2B-UNet**: 3D UNet model enhanced with attention blocks.
2. **AR2B-DeepSup-UNet**: Base model with deep supervision.
3. **Swin-AR2B-DeepSup-UNet**: Integrates Swin Transformers into the UNet encoder.

**Evaluation:** Rigorous training and validation with a focus on the benefits of attention, deep supervision, and multi-objective loss.
**Performance:** Achieved a high mean Dice score, indicating effective segmentation capabilities.
**Contributions:**
This work contributes to the field of medical image analysis by demonstrating the effective integration of contemporary techniques in CNN architectures for brain tumor segmentation. It highlights the potential and limitations of these approaches, providing a foundation for future research and development in automated medical image segmentation.

This README provides an overview of our dissertation project, "Enhancing Brain Tumor Segmentation in Multimodal MRI Scans," and serves as a guide for the attached source code. This folder contains project code, a sampled dataset, and related resources for a medical image segmentation project employing various deep learning models. Below, you'll find detailed information about the contents of the `src` folder, the structure of the `data` folder, and the `saved_models` folder.

## Contents of the `src` Folder

### Data Download and Preprocessing

1. `00_data_download.py`: Downloads the BraTS 2023 training dataset using SynapseClient and extracts it to the `../input` folder.

2. `01_create_directories.py`: Creates the necessary directory structure for both the nnUNet framework and our custom models.

3. `02_data_description.ipynb`: Provides a comprehensive description of the dataset, checking for missing files, analyzing segmentation labels, and visualizing original and cropped modalities.

4. `02_data_nnunet_split.py`: Splits the input dataset into training and validation sets in the format required for nnUNetv2.

5. `03_data_nnunet_to_npy.py`: Preprocesses and converts the training and test dataset, previously split for nnUNet, into a suitable format for our custom models.

### Resources

6. `custom_data_generator.py`: Implements a custom dataloader in PyTorch to efficiently feed data in batches to our custom models.

7. `losses.py`: Contains implementations of various loss functions and evaluation metrics for model training and evaluation.

8. `models.py`: Implements key architectural blocks and model architectures for AR2B, AR2B-DeepSup, and Swin-AR2B-DeepSup.

9. `generate_predictions.py`: Generates predictions from our custom models and provides corresponding ground truth data for evaluation.

10. `nnunet_evaluation.py`: Evaluates the predictions made by nnUNetv2, utilizing predefined evaluation metrics.

11. `write_options.py`: Implements a DualStream module to direct the output simultaneously to the screen and an output file, aiding in better management of results and logs.

12. `plots.ipynb`: Provides code to plot and annotate model performances from our evaluation and experiments.

13. `qualitative_visualisation.ipynb`: Visualizes the predictions of the models against the ground truth.

### Model Training

14. `17_training_base_parallel.py`: Trains the base model AR2B using combined loss with class weighting.

15. `08_training_deep_parallel.py`: Trains the deep supervision model AR2B-DeepSup using combined loss with class weighting.

16. `09_training_swin_parallel.py`: Trains the Swin transformer model Swin-AR2B-DeepSup, utilizing combined loss with class weighting.

### Experiments

17. `07_training_base_parallel_cross_entropy.py`: Trains the base model AR2B using cross-entropy loss with class weighting as an alternative approach.

18. `28_training_deep_parallel_learning_rate.py`: Conducts experiments with different learning rates for training AR2B-DeepSup.

19. `38_training_deep_parallel_equal_weights.py`: Trains the deep supervision model AR2B-DeepSup using combined loss with equal weights for the background and tumor classes.

## Folder Structure in the `data` Folder

The `data` folder is structured as follows:

    * data
        o training
            * images
            * masks
            * nnUNet_raw
                * Dataset001_BRATS
                    o imagesTr
                    o labelsTr
                * nnUNet_preprocessed
                * nnUNet_results
        o test
            * images
            * masks
            * nnUNet_raw
                * Dataset001_BRATS
                    o imagesTs
                    o labelsTs
            * nnUNet_results


This organized structure effectively manages the training and test data for the project.

### Contents of the `data` Folder

The `data` folder contains a total of 30 training records, including both images and masks, along with 8 records designated for validation and testing. These samples are preprocessed and formatted for training and evaluating the models.

To download the raw dataset or if the dataset file is missing, please run the `00_data_download.py` script and follow the data preprocessing scripts sequentially.

## Additional Details Regarding the Attached Zip File

### Contents of the `saved_models` Folder

The `saved_models` folder contains best model checkpoints from our training:

- `17_best_model_base.pt`: The best AR2B model trained on Combined Loss.
- `08_best_model_deep.pt`: The best AR2B-DeepSup model trained on Combined Loss.
- `09_best_model_swin.pt`: The best Swin-AR2B-DeepSup model trained on Combined Loss.

These saved models can be used for inference or further experimentation.

### Contents of the `predictions` Folder

In the `predictions` folder, we have saved ground truths and model predictions from our three models for 2 randomly selected cases. You can visualize them using the `qualitative_visualisation.ipynb` notebook.

Thank you for using our project code!
