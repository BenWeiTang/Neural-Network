# Neural-Network
## Overview
This project provides a configurable neural network written in C. The user can choose the number of input neurons, output neurons, number of hidden layers, and the number of neurons for each individual layers. The project was developed to use the MNIST handwritten digit training data. However, with some tweaks, the functionality can be expanded to use other training data that have been vectorized into normalized floating point variables.

## Configuring the Neural Network
In `main.c`, you can see the configuration is done with an int array `config`. The first element specifies the number of input neurons, the second specifies the number of hidden layers, and the rest tell the number of nerons for each layer. **Note that the last hidden layer is considered the output layer.**

You can also see that the function `makeNeuralNetwork` takes in two arguments. The first is a pointer to the `config` array, and the second specifies the length of the array. 
It is **important** that the `config` array makes sense in that the second int in the array should match the number of the rest elements. It is also **important** that the second argument of `makeNeuralNetwork` (the length of the array as the first argument) match the actual length of the `config` array that is passed in.

## Training data
The project was developed with the MNIST handwritten digit training data in mind. This is evident in the test and image scripts where the training data should be in the csv format and be put inside a `/data` directory, which needs to be created by the user. Due to their sizes, it is not possible to conclude those training files on Github, but they can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
