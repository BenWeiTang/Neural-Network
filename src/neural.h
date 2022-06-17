#ifndef NEURAL_H
#define NEURAL_H

#include <math.h>
#include "../src/matrix.h"

typedef struct
{
    int numNeuron;
    Matrix* w;
    Matrix* bias;
    Matrix* z;
    Matrix* activations;
} Layer;

typedef struct
{
    int numInput;
    int numHiddenLayer;
    Layer** layers;
    Matrix* inputs;
} NeuralNetwork;

Layer* makeLayer(int curNeuCount, int prevNeuCount);
NeuralNetwork* makeNeuralNetwork(int* config, int configLen);
void deleteLayer(Layer* layer);
void deleteNeuralNetwork(NeuralNetwork* NN);
void feedForward(NeuralNetwork* NN, double* inputs);
void predict(NeuralNetwork* NN, double* inputs);
void backPropogate(NeuralNetwork* NN, double* inputs, double* observedValues);

#endif