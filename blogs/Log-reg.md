# Logistic Regression as a Neural Network

## Binary Classification

We will use logistic regression for a classification task to illustrate how a NN works and highlight interesting interesting differences between the classical algorithm and the NN version.

 For instance we have an image of a dog and our task is to represent if the image contains a dog or not 1 -> cat 0 otherwise. recall that a computer represents image as three separate matrices corresponding to the 3 (RGB) values. So if our image has dimensions `32 x 32` then we will have 3 (32 x 32) matrices of pixel intensity values for the image. Computers cannot will not be able to process these matrices as they are so we turn them into `feature vectors` by unrolling them into one long vector so it becomes a vector of size `32 x 32 x 3`

## Log reg basics

Given an input feature vector X which is for instance the image of a dog, an algorithm that can out a prediction.

To train the parameters we need to define a cost function, formally given that we have a training set of data we want a model so that the outputs of the training set match the ground truth.

We also need to use a `Loss Function` -(ylog\bar{y}+(1-y)log(1-\bar{y})) to measure how well our model is doing. `Cost function` tells us how well our model is doing on the entire training set.

`Gradient Descent` is used to learn the parameters `w` and `b` which minimize J(w,b). The cost function looks like a bowl which is convex. We initialize the parameters to a certain value and at each step.

The operations of a NN consist of a `forward pass` where we compute the outputs of the NN and then a `Backward pass` where it calculates the parameters  or derivatives. The `Computation Graph` helps us understand why NNs work this way. Let's say we have a function of J(a,b,c) = 3(a+bc) it consists of 3 steps:
1. u = bc
2. v = a + u
3. J = 3v

In context of our problem the J is our cost function which we want to optimize. Left to right pass gives the value of the function. To learn the parameters w, b meaning we want to minimize J we `backpropagate` the errors to see how much we need to tweak the parameters.

## Unoptimized Gradient Descent
J = 0, dw = 0, dw2 = 0, db = 0
for i to m
  z(i) = wTX+b
  a(i) = sigmoid(z(i))
  J += -[y(i)loga(i)+(1-y(i)log(1-a(i)))]
    dz(i) += X(i)dz(i)
    dw += X_1(i)dz(i)
    dw2 += X_2(i)dz(i)
    db += dz(i) 
  J /= m
  dw /= m; dw2 /= m; db /= m
  w = w - alpha*dw
  w2 = w2 - alpha*dw2
  b = b - alpha*db
