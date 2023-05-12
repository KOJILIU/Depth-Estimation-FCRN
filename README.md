# Depth-Estimation-experiment
This is the repo for DS-UA 301 Final Project

The README file should include information on the project's structure, software requirements, and instructions for training and testing the models
Include performance evaluation results (MSE Loss) for the CNN and FCRN models on the NYU Depth V2 and Apolloscape datasets

Objectives of this experiment:
The objective of this project is to develop a neural network algorithm that can take a monocular image and produce depth values predictions for each image. This algorithm aims to help individuals with visual impairments navigate their surroundings more effectively with other assistive devices. The given depth predictions will be mapped to a color map. For instance, pixels that have smaller predicted depth values will be assigned to smaller distances. Additionally, if we have more time, we might create a soft robotic vest that will allow individuals to sense their surroundings through tactile sensory information. The vest will convert spatial data from the environment into tactile sensations that project onto the patient's body.


We will use a CNN (Convolutional Neural Network) network to develop the video dimension measurement algorithm. CNNs are a type of deep learning neural network that are well-suited for image and video processing tasks. We will start with a local-based framework for CNN training and testing. We will also use an existing dataset to train the network. Our goal is to use FCRN (Fully Convolutional Residual Networks) to give depth values predictions. The model is built on ResNet-50 and fully convolutional network architecture, combined with up-sampling blocks to give higher resolutions with less parameters. As for optimization, we will use MSE loss to evaluate the performance.

