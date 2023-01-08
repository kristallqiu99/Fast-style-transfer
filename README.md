# Neural Style Transfer
This project is in fulfillment of CSCI-SHU 360 Machine Learning in Spring 2021 at NYU Shanghai.

Group member: Xinyue Liu, Yuejiao Qiu

## Problem description
The development of convolutional neural networks (CNNs) has enabled the computer to extract abstract features of images, which also leads to the emergence of deep-learning-driven image modification, <i>neural style transfer</i>. Neural style transfer consists of synthesising a texture from a source image while constraining the texture synthesis in order to preserve the content of a target image. In this project, we implement two different algorithms to realize the artistic style transformation. We first use VGG19 for a fixed pairwise  style and content in an iteration approach, and then adopt image transfer network to train a sytle-specific model for fast style transfer.

## Data
As for style images, we primarily use famous paintings as styles, which are accessible at [Style](/Style). The original source is from [Kaggle](https://www.kaggle.com/momincks/paintings-for-artistic-style-transfer).

The first model, basic style transfer using IOB-NST algorithms, inputs one specific style image and one content image into the iteration loop. Therefore, in the first model, we can use an arbitrary pair of style and content images for each transfer. However, for the second generative model, since we are going to train the model for real-time style transfer, it is of great importance to use a sufficiently large dataset as content images to train the model. Therefore, we choose to use [LabelMe](http://labelme2.csail.mit.edu/Release3.0/browserTools/php/publications.php), which is a large dataset created by the MIT Computer Science and Artificial Intelligence Laboratory containing 187,240 images. For actual training, due to time and efficiency concern, we only use a subset of 20,000 images.

## Experiment
### Basic style transfer
Basic style transfer ultilizes a pre-trained VGG19 as the loss network, which includes 16 convolutional and 5 pooling layers. We normalize the network by scaling the weights so that the mean activation of each filter over images and positions is equal to one. And we replace max pooling by average pooling. We also remove the fully connected layer, since we can directly output an image of the same dimensions. According to Gatys et al., we will use {relu1_1, relu2_1, relu3_1, relu4_1, relu5_1} as style layers and {relu4_2} as the content layer. The results are as below.
![Alt text](Output/S1-output.png)

### Fast style transfer
Although the previously-discussed IOB-NST is able to yield pretty impressive results, the efficiency is a big issue because we need to input a new image and go over the iterations every time if we want to use either a new style or content image. Therefore, we build a model to perform real-time style transfer using TransformNet.

The system is as follows. We train an image transformation network to transform input images into output images, and  use a loss network pre-trained, i.e. VGG19, to define perceptual loss functions that measure perceptual differences in content and style between images. The loss network remains fixed during the training process. Detailed architechture is described in the report. The results are shown below, which include multiple fixed style images to variable  content images.
![Alt text](Output/S2-output.png)

## Summary
Detailed summary of this project is included in [ML_Final_Paper.pdf](ML_Final_Paper.pdf)
