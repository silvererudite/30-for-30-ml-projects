# Logistic Regression as a Neural Network

## Binary Classification

We will use logistic regression for a classification task to illustrate how a NN works and highlight interesting interesting differences between the classical algorithm and the NN version.

 For instance we have an image of a dog and our task is to represent if the image contains a dog or not 1 -> cat 0 otherwise. recall that a computer represents image as three separate matrices corresponding to the 3 (RGB) values. So if our image has dimensions `32 x 32` then we will have 3 (32 x 32) matrices of pixel intensity values for the image. Computers cannot will not be able to process these matrices as they are so we turn them into `feature vectors` by unrolling them into one long vector so it becomes a vector of size `32 x 32 x 3`

## Log reg basics

Given an input feature vector X which is for instance the image of a dog, an algorithm that can out a prediction.

To train the parameters we need to define a cost function, formally given that we have a training set of data we want a model so that the outputs of the training set match the ground truth.

We also need to use a `Loss Function` to measure how well our model is doing.
