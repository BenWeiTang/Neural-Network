# Neural-Network
## Overview
This project provides a configurable neural network that is written in C. The user can configure the number of input neurons, output neurons, number of hidden layers, and the number of neurons for each individual layers.
The project was developed with the MNIST Handwritten digit training data in mind. This is evident in the test and image scripts where the training data should be in the csv format and be put inside a "/data" directory, which should be created by the user. Due to the size of the files, it is not possible to be concluded on Github, but it can be found at: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv 

## Configuring the neural network
In main.c, you can see the configuration is done with an int array. The first element specifies the number of input neurons, the second specifies the number of hidden layers, and the rest specifies the number of nerons for each layer.
You can also see that the function makeNeuralNetwork takes in two arguments. The first is pointer to the array of configuration, and the second specifies the length of the array. It is **important** that the configuration array makes sense as well as that the second argument of makeNeuralNetwork (length of the array) match up the actual length of the config array.

## Training data
