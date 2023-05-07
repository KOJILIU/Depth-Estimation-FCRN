# Depth-Estimation-experiment
This is the repo for DS-UA 301 Final Project

Objectives of this experiment:
The objective of this project is to develop a neural network algorithm that can take a monocular image and produce depth values predictions for each image. This algorithm aims to help individuals with visual impairments navigate their surroundings more effectively with other assistive devices. The given depth predictions will be mapped to a color map. For instance, pixels that have smaller predicted depth values will be assigned to smaller distances. Additionally, if we have more time, we might create a soft robotic vest that will allow individuals to sense their surroundings through tactile sensory information. The vest will convert spatial data from the environment into tactile sensations that project onto the patient's body.

4. Challenges
One of the main challenges of this project will be identifying the difference of distance of pixels that are not consistent in a single image. In other words, if the objects are distant in a monocular image, such as a person standing in front of the camera and a mountain from afar, then the algorithm might find it difficult to precisely predict the distance. In addition, since we are trying to find another way besides using hardware distance estimation techniques, such as LiDAR, the algorithm must be accurate enough to ensure that it does not incorrectly identify pixels as either too close or too far away. Notice that the performance of the algorithm might not be as good as hardware, but it is more efficient in terms of energy.
5. Approach/Techniques
We will use a CNN (Convolutional Neural Network) network to develop the video dimension measurement algorithm. CNNs are a type of deep learning neural network that are well-suited for image and video processing tasks. We will start with a local-based framework for CNN training and testing. We will also use an existing dataset to train the network. Our goal is to use FCRN (Fully Convolutional Residual Networks) to give depth values predictions. The model is built on ResNet-50 and fully convolutional network architecture, combined with up-sampling blocks to give higher resolutions with less parameters. As for optimization, we will use Huber loss 
6. Implementation details: hardware (type of compute GPU/TPU etc, cloud based, edge devices), software (framework, existing code to reuse), dataset
We will mainly use Google Colab’s GPU as well as local devices, such as RTX 2080 and RTX 3070. Our neural network will be based upon ResNet-50 with up-sampling blocks and NYU Depth v2 benchmark dataset.
7. Demo planned
We plan to demonstrate a live video dimension measurement algorithm in action, showing how it marks close and far objects in a video. We will also show how this algorithm can be useful for individuals with visual impairments.
8. References (if any)
https://github.com/irolaina/FCRN-DepthPrediction
https://arxiv.org/abs/1606.00373 (Deeper Depth Prediction with Fully Convolutional Residual Networks)
