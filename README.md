# Image Classification with Convolutional Neural Networks

This project aims to classify images using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. The model is implemented using the PyTorch library and incorporates techniques such as data augmentation, dropout, and batch normalization to improve performance and prevent overfitting.

## Project Overview

- **Institution**: Faculty of Electrical and Computer Engineering, Cracow University of Technology

## Introduction

The project involves building and training a CNN model to classify images from the CIFAR-10 dataset, which consists of 60,000 color images in 10 classes, with 6,000 images per class. The model's performance is evaluated based on validation loss and classification accuracy on the test set.

## Theoretical Background

### Convolutional Neural Network (CNN)

CNNs are specialized architectures for processing data with a spatial structure, such as images. Key layers in CNNs include:
- **Convolutional Layer**: Applies filters to the input to extract features like edges and textures.
- **ReLU Layer**: Introduces non-linearity by converting negative values to zero.
- **Pooling Layer**: Reduces the dimensions of the data to improve computational efficiency.
- **Fully Connected Layer**: Transforms spatial data into a vector for classification.

## Project Description

### Dataset

The CIFAR-10 dataset includes images from the following classes:
- Airplanes
- Cars
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks

### Methodology

The project uses the following techniques:
- **Data Augmentation**: Increases the diversity of training data by applying random transformations like rotations and flips.
- **Dropout**: Regularization technique that randomly sets some neuron outputs to zero during training to prevent overfitting.
- **Batch Normalization**: Normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

### Model Architecture

The CNN model consists of:
- Five convolutional layers with batch normalization and ReLU activation.
- Max pooling layers to reduce spatial dimensions.
- Fully connected layers for classification.
- Dropout layers to prevent overfitting.

### Training

The model is trained using the following steps:
1. **Data Preparation**: Load and augment the CIFAR-10 dataset.
2. **Model Definition**: Define the CNN architecture using PyTorch.
3. **Loss and Optimizer**: Use CrossEntropyLoss and SGD optimizer with learning rate scheduling.
4. **Training Loop**: Train the model over multiple epochs, validating on a separate validation set to monitor performance.

### Evaluation

The model's performance is evaluated using:
- **Training Loss**: Indicates how well the model is learning the training data.
- **Validation Loss**: Indicates how well the model generalizes to unseen data.
- **Accuracy**: Percentage of correctly classified images on the test set.
- **Confusion Matrix**: Visual representation of model performance across different classes.

## Results

The model achieves promising results, showing a steady increase in accuracy and a decrease in loss over the training epochs. Further training may improve performance.

