# CNN-implementation
# CNN and ResNet18 for Smoking Classification

## Project Overview
This project consists of two parts, each focused on binary classification of smoking vs. non-smoking images:
1. **Custom CNN Model** – A convolutional neural network (CNN) built from scratch.
2. **ResNet18 Pretrained Model** – Using the ResNet18 architecture for comparison.

Both models were evaluated on two different datasets:
- **Dataset 1 (Regular Set)**: A relatively standard image set for classification.
- **Dataset 2 (Harder Set)**: A much more challenging image set for classification.

### Data Augmentation
During the first phase of testing, I created an **intermediate dataset** by applying **data augmentation** to Dataset 1. This generated an extended training set that served as a difficulty transition between the two datasets.

### Research Objective
- Compare the performance of the **CNN model** and **ResNet18** on Dataset 1.
- Evaluate the models on the **augmented dataset** to understand the impact of data augmentation.
- Assess both models on **Dataset 2** to analyze their robustness to more challenging data.

# My Network Description
A custom CNN was designed to classify images into two categories: smoking (1) and non-smoking (0). The architecture includes:

### Convolutional Layers
- **Three convolutional layers** with ReLU activation to extract image features.
- Each layer uses **32, 64, and 128 filters** of size **3×3** with `padding=1` to maintain spatial dimensions.

### MaxPooling Layers
- **MaxPooling (2×2)** after each convolutional layer to reduce spatial dimensions by half.

### Fully Connected (Dense) Layers
- **First fully connected layer**: 128 neurons for deeper representation learning.
- **Output layer**: Single neuron with **Sigmoid activation** for binary classification.

### Regularization - Dropout - was changed during the experiance 
- Applied **Dropout (0.5)** to reduce overfitting. 

### Training Parameters - ware changed during the experiance 
- **Learning rate**: 0.001
- **Epochs**: 10 (due to CPU constraints and insights from previous experiments)
- **Batch size**: 32

## Image Processing
- **Resized images** to **256×256** before feeding into the model.
- After three convolutional layers with MaxPooling, feature map size is **32×32**.
- The final **fully connected layer** has **131,072 neurons (32×32×128)**.

## Transform Function
- **Resizing images** to 256×256.
- **Converting to tensors**.
- **Normalizing pixel values** to **[-1,1]**.
- **Data augmentation** using PyTorch's `RandomRotation` to introduce variation when necessary.

## Validation Strategy
Within the `train_model` function, the model was validated using a separate validation set (`X_val`, `y_val`) after each epoch to:
1. Evaluate generalization beyond `X_train`.
2. Detect signs of overfitting (if validation accuracy is significantly lower than training accuracy).
3. Confirm that improvements in `X_train` also reflect improvements in `X_val`.

## Results & Key Insights
- Data augmentation improved generalization and robustness.
- Comparing CNN and ResNet18 on different datasets revealed differences in how each model adapts to varying difficulty levels.
- **Dataset 2** posed a significant challenge, highlighting the strengths and weaknesses of both models.

---
This project provides insights into how deep learning models behave under different data conditions and the impact of architectural choices on classification performance.
