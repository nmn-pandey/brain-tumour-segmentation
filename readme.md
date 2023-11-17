# Enhancing Brain Tumor Segmentation in Multi-modal MRI Images

This repository contains the code implementation of my Master's dissertation project at Brunel University London. The project focuses on developing advanced methods for automated segmentation of brain tumors from multimodal MRI scans by integrating modern mechanisms like Attention, Multi-Objective Losses and Swin Transformers into the traditional convolutional neural networks (CNNs) based U-Net architecture. Key challenges addressed include localizing indistinct tumor boundaries, handling class imbalances in medical datasets, and efficiently capturing global context. This work integrates attention mechanisms, multi-scale deep supervision, multi-objective loss functions, and Swin-Transformers into CNN-based architectures.

In this README file, we provide an overview and explanation of the code and models developed for this project. For a comprehensive understanding of the methodologies, theories, and detailed analysis, we highly recommend referring to the code overview section at the end of this file and the comments in the code itself. If you require further help or would like to take this study forward, I would be pleased to support you.

### Dataset
The Brain Tumor Segmentation (BraTS) 2023 dataset with T1, T1-contrasted, T2, and FLAIR MRI scans, including expert annotations for 1251 cases.
![image](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/0def1930-8d01-449b-a939-26f9657e5325)

### Model Architectures
1. **AR2B-UNet**: A 3D UNet Model enhanced with Attention Blocks. This architecture is designed to improve the segmentation accuracy by focusing on relevant features in the MRI scans.  
![AR2B-UNet-1](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/0bef9e31-ba8b-456c-ab8b-04dd92edf555)

2. **AR2B-DeepSup-UNet**: An extension of the base model (AR2B-UNet) enhanced with Deep Supervision. Deep Supervision facilitates the training of deeper networks by addressing the vanishing gradient problem and improving feature learning at multiple levels.
![AR2B-DeepSup-UNet-1](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/a3704f28-0838-43f2-a9ec-a1d7e693a8b3)

3. **Swin-AR2B-DeepSup-UNet**: This model integrates Swin Transformers into the UNet Encoder. The Swin Transformer is designed to capture global context more effectively, which is crucial for accurate segmentation in complex medical images like MRI scans.
![Swin-AR2B-DeepSup-UNet-1](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/ce26905f-3bb5-4f0e-b2f6-d0c9c97c9e14)

### Key Architectural Blocks
1. Convolutional, and Feature Refinement Blocks to extract hierarchical features from the input.
2. Attention Blocks to focus on the relevant features before passing them on to the decoder.
3. Swin Transformer Block for capturing global context.
4. Max Pooling and Transpose Convolution Layers for downsampling and upsampling the feature maps, respectively.
5. 3D Convolution Layers for Interpolation.

### Evaluation
Rigorous training and validation with a focus on the benefits of attention, deep supervision, and multi-objective loss, was performed. The performance of the three models, AR2B, AR2B-DeepSup, and Swin-AR2B-DeepSup, was evaluated based on several key metrics, including Accuracy, Loss, Dice Coefficients, Sensitivity, Specificity, and Precision.

These metrics were computed for each tumor class during both training and validation phases. The highest mean Dice Coefficient achieved during these phases was used to select the best epoch for each model.

### Performance

1. AR2B Model: Achieved the minimum training loss of 0.145 but displayed occasional spikes indicating potential training instability. It achieved a maximum validation dice of 0.691 at Epoch 350, which then decreased and the training loss increased, suggesting overfitting.
2. AR2B-DeepSup Model: Maintained more stable training with fewer spikes and achieved minimum training losses of 0.330. During validation, this model surpassed the others, recording a mean Dice score of 0.795 at epoch 900.
3. Swin-AR2B-DeepSup Model: Achieved minimum training losses of 0.533 and during validation reached a mean Dice score of 0.757 at epoch 950.

![training and validation dice](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/8aff2a91-e945-4717-810e-ecd732931011)

**Table comparing the model performance**

| Models | Dice Coefficient (1) | (2) | (3) | Mean | Sensitivity (1) | (2) | (3) | Mean | Specificity (1) | (2) | (3) | Mean |
|--------|----------------------|-----|-----|------|-----------------|-----|-----|------|-----------------|-----|-----|------|
| AR2B   | 0.650                | 0.653 | 0.768 | 0.691 | 0.582         | 0.842 | 0.824 | 0.749 | 0.999         | 0.972 | 0.995 | 0.989 |
| AR2B-DeepSup | **0.770 **         | 0.770 | 0.843 | 0.795 | **0.800 **        | 0.843 | 0.911 | 0.851 | 0.997         | 0.987 | 0.998 | 0.994 |
| Swin-AR2B-DeepSup | 0.729     | 0.736 | 0.807 | 0.757 | 0.685         | 0.827 | 0.896 | 0.802 | 0.933         | 0.999 | 0.988 | 0.973 |
| nnU-Netv2 | 0.756             | 0.861 | 0.850 | 0.822 | 0.763         | 0.893 | 0.926 | 0.873 | 1.000         | 0.999 | 1.000 | 1.000 |

The AR2B-DeepSup model achieved the highest overall performance among our developed models across all evaluated metrics​, and even surpasses nnU-Netv2 performance for tumour class 1.

### Qualitative Results
#### Accurate Predictions by All Models:

All models demonstrated high overlap with the ground truth segmentation in well-segmented cases, indicating accurate segmentation capabilities under certain conditions.

![all](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/e6806e5c-7199-4e29-a353-0c68be77cf3e)

#### Superior Performance of AR2B-DeepSup:

This model most closely matched the true tumor shape and outperformed the other two models, capturing intricate tumor morphology, especially for Tumor Class 1 – the NCR region.

![ar2b_deepsup](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/89c6e6ca-34c3-4323-88a1-45a2e1d9fe79)

#### Better Background Identification by Swin-AR2B-DeepSup:

In some cases, this model outperformed the others by better identifying the background region and more accurately predicting tumor boundaries.

![swin](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/38871506-0a11-41f8-92b0-255b3f1285c7)

#### Challenges in Poorly-Segmented Cases:

All models struggled in some cases, especially those with low contrast, failing to detect tumors or making false positive predictions, indicating room for performance improvement on challenging cases​​.

![none](https://github.com/nmn-pandey/brain-tumour-segmentation/assets/20767834/35e461c6-d0ea-4376-b50d-fd3534a441cd)


### Contributions:
This work contributes to the field of medical image analysis by demonstrating the effective integration of contemporary techniques in CNN architectures for brain tumor segmentation. It highlights the potential and limitations of these approaches, providing a foundation for future research and development in automated medical image segmentation.

## Code Overview
The below section provides an overview of my dissertation project, "Enhancing Brain Tumor Segmentation in Multimodal MRI Scans," and serves as a guide for the attached source code. This folder contains project code, a sampled dataset, and related resources for a medical image segmentation project employing various deep learning models. Below, you'll find detailed information about the contents of the `src` folder, the structure of the `data` folder, and the `saved_models` folder.

## Contents of the `src` Folder

### Data Download and Preprocessing

1. `00_data_download.py`: Downloads the BraTS 2023 training dataset using SynapseClient and extracts it to the `../input` folder. **Please edit the username and password with your credentials from the Synapse website to download the dataset.**

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

To download the raw dataset or if the dataset file is missing, please run the `00_data_download.py` script with your credentials and follow the data preprocessing scripts sequentially.

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
