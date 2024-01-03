# MNIST_Neural_Networks

This repository contains a series of projects exploring different neural network architectures applied to the MNIST and Fashion MNIST datasets. These projects demonstrate a progression from basic neural networks to more complex Convolutional Neural Networks (CNNs), showcasing various techniques and approaches in machine learning.

## Projects Overview

### Basic Neural Network for MNIST (mnistTrain.py):
- A simple feedforward neural network implementation.
- Designed to recognize handwritten digits from the MNIST dataset.
- Features customizable activation functions.

### CNN for MNIST (cnn_mnist.py):
- An implementation of a Convolutional Neural Network.
- Tailored for image recognition tasks with the MNIST dataset.
- Utilizes layers like Conv2D, MaxPooling2D, Dropout, and Flatten.
- CNN for Fashion MNIST with Image Augmentation (mnist_fashion_CNN.py):

### A CNN adapted for the Fashion MNIST dataset.
- Incorporates ImageDataGenerator for image augmentation.
- Demonstrates the application of CNNs to fashion item recognition.

## Installation and Usage
To run these projects, you need Python and the necessary libraries, including TensorFlow, Keras, Numpy, and Scipy. You can install these dependencies using:

pip install tensorflow keras numpy scipy

To execute a script, navigate to the script's directory and run:

python script_name.py

Replace script_name.py with the name of the script you want to run (e.g., mnistTrain.py).

Alternatively can be run from your chosen IDE. 


# Convolutional Neural Networks on MNIST
This project showcases the application of Convolutional Neural Networks (CNNs), a powerful type of neural network particularly effective for image recognition tasks, to the MNIST dataset.

## Understanding CNNs
CNNs are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are particularly known for their ability to pick up on spatial hierarchies in images by applying convolutional layers, which filter inputs for useful information. These layers effectively reduce the number of parameters, enabling the network to be deeper with fewer parameters.

## Key Components of CNNs:
- Convolutional Layers: Apply a number of filters to the input. Each filter activates certain features from the input.
- Pooling Layers: Reduce the spatial size (width and height) of the input volume for the next convolutional layer. They are used to decrease the computational power required and to reduce overfitting.
- Dense (Fully Connected) Layers: These layers are typical neural network layers where all neurons from the previous layer are connected to each neuron.
- CNNs are particularly useful for image classification tasks because they can learn and identify spatial hierarchies in an image, such as lines and curves, and then learn to combine these elements into larger structures (like shapes) and eventually into a complete object (like a handwritten digit).

## Project Description
This project includes a script to apply a CNN to the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems. The script mnist_CNN.py and mnist_fashion_CNN applies a CNN to the MNIST datasets, a large database of handwritten digits or various clothing items commonly used for training various image processing systems. These scripts involve the following steps:

- Loading and preprocessing the MNIST dataset.
- Defining a CNN model architecture using TensorFlow and Keras.
- Training the model on the MNIST dataset.
- Evaluating the model's performance.

## Insights and Observations
### Summary of Results
The Convolutional Neural Network (CNN) model trained on the MNIST dataset achieved >99% accuracy and >90% accuracy on the fashion MNST dataset. This high level of accuracy demonstrates the effectiveness of CNNs in image classification tasks, especially in recognizing patterns and features in images like handwritten digits.

#### Interesting Findings
- Feature Learning: The CNN's ability to learn features automatically from the data, as opposed to manual feature extraction, was particularly notable. It could identify intricate patterns in the digits that are crucial for classification.
- Impact of Layer Depth: Experimentation with different numbers of convolutional layers showed that deeper models could capture more complex features but also required more data and computational resources to avoid overfitting.
#### Challenges Faced
- Model Tuning: One of the main challenges was finding the right balance in the model's architecture—too few layers and the network couldn't learn complex patterns, too many and it started overfitting.
- Computational Constraints: Training deeper CNN models significantly increased computational load, necessitating a balance between model complexity and available resources.
Learnings from the Project
- Importance of Data Preprocessing: Effective normalization and preprocessing of image data were crucial for the model's performance.
- Layer Experimentation: Experimenting with different types of layers and their configurations (like the number of filters in convolutional layers) was key to optimizing the CNN.

#### Comparison with Other Techniques
- Compared to simpler machine learning models like SVM or k-Nearest Neighbors, the CNN showed superior performance in handling raw image data.
- Unlike traditional algorithms, CNNs eliminate the need for manual feature extraction, making them more scalable and efficient for complex image classification tasks.

Please see my other project 'MNIST_PCA_kNN' for another machine learning approach to catagorise the MNIST datasets.
