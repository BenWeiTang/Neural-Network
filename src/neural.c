#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/neural.h"
#include "../src/matrix.h"
#include "../src/activation.h"

Layer* makeLayer(int curNeuCount, int prevNeuCount)
{
    Layer* result = malloc(sizeof(Layer));
    result->numNeuron = curNeuCount;
    result->w = makeMatrix(curNeuCount, prevNeuCount);
    result->bias = makeMatrix(curNeuCount, 1);
    result->z = makeMatrix(curNeuCount, 1);
    result->activations = makeMatrix(curNeuCount, 1);
    matrixRandomize(result->w);
    return result;
}

NeuralNetwork* makeNeuralNetwork(int* config, int configLen)
{
    NeuralNetwork* result = malloc(sizeof(NeuralNetwork));
    result->numInput = config[0];
    result->numOutput = config[configLen-1]; // output will match last hidden layer
    result->numHiddenLayer = config[1];
    result->layers = malloc(sizeof(Layer*) * result->numHiddenLayer);
    result->inputs = makeMatrix(result->numInput, 1);
    result->outputs = makeMatrix(result->numOutput, 1);

    // Starting from the 3rd element in config, specifies the num of neurons per layer
    for (int i = 2; i < configLen; i++)
    {
        int curNeuCount = config[i];
        int prevNeuCount = i == 2 ? config[0] : config[i-1]; // For first hidden layer, manually grab numInput as the prevNeuCount
        result->layers[i-2] = makeLayer(curNeuCount, prevNeuCount); // First hidden layer starts at index 0
    }
    return result;
}

void deleteLayer(Layer* layer)
{
    deleteMatrix(layer->w);
    deleteMatrix(layer->bias);
    deleteMatrix(layer->z);
    deleteMatrix(layer->activations);
    free(layer);
    layer = NULL;
}

void deleteNeuralNetwork(NeuralNetwork* NN)
{
    for (int i = 0; i < NN->numHiddenLayer; i++)
    {
        deleteLayer(NN->layers[i]);
    }
    free(NN->layers);
    free(NN->inputs);
    free(NN->outputs);
    free(NN);
    NN = NULL;
}

void feedForward(NeuralNetwork* NN, double* inputs)
{
    for (int i = 0; i < NN->numInput; i++)
    {
        NN->inputs->data[i][0] = inputs[i];
    }

    for (int i = 0; i < NN->numHiddenLayer; i++)
    {
        deleteMatrix(NN->layers[i]->activations);
        deleteMatrix(NN->layers[i]->z);
        Matrix* zPreBias = matrixMul(NN->layers[i]->w, i == 0 ? NN->inputs : NN->layers[i-1]->activations);
        NN->layers[i]->z = matrixAdd(zPreBias, NN->layers[i]->bias);
        NN->layers[i]->activations = copyMatrix(NN->layers[i]->z);
        matrixApply(NN->layers[i]->activations, &lrelu);
        deleteMatrix(zPreBias);
    }
    deleteMatrix(NN->outputs);
    NN->outputs = argMax(NN->layers[NN->numHiddenLayer-1]->activations);
}

void backPropogate(NeuralNetwork* NN, double* inputs, double* observedValues)
{
    // Updtate the outputs in NN
    feedForward(NN, inputs);

    // Copy array into matrix
    Matrix* observed = makeMatrix(NN->numOutput, 1);
    for (int r = 0; r < NN->numOutput; r++)
    {
        observed->data[r][0] = observedValues[r];
    }

    // Compute error of the output layer
    Matrix** layerErrors = malloc(sizeof(Matrix*) * NN->numHiddenLayer);
    Matrix* dCost = matrixSub(NN->layers[NN->numHiddenLayer-1]->activations, observed);
    Matrix* dLrelu = copyMatrix(NN->layers[NN->numHiddenLayer-1]->z);
    matrixApply(dLrelu, &dlrelu);
    layerErrors[NN->numHiddenLayer-1] = matrixHadaMul(dCost, dLrelu); // output layer error

    deleteMatrix(observed);
    deleteMatrix(dCost);
    deleteMatrix(dLrelu);

    // Back propogate
    for (int l = NN->numHiddenLayer-2; l >= 0; l--)
    {
        Matrix* previousT = matrixTranspose(NN->layers[l+1]->w);
        Matrix* previousError = matrixMul(previousT, layerErrors[l+1]);
        Matrix* dLrelu = copyMatrix(NN->layers[l]->z);
        matrixApply(dLrelu, &dlrelu);
        layerErrors[l] = matrixHadaMul(previousError, dLrelu);

        deleteMatrix(previousT);
        deleteMatrix(previousError);
        deleteMatrix(dLrelu);
    }

    // Calculate gradients for weights and biases
    for (int l = 0; l < NN->numHiddenLayer; l++)
    {
        Matrix* prevActT = matrixTranspose(l == 0 ? NN->inputs : NN->layers[l-1]->activations);
        Matrix* deltas = matrixMul(layerErrors[l], prevActT);
        
        // In-place update to current layer's weights
        for (int r = 0; r < deltas->row; r++)
        {
            for (int c = 0; c < deltas->col; c++)
            {
                NN->layers[l]->w->data[r][c] -= (0.01 * (deltas->data[r][c]));
            }
        }

        // In-place update to current layer's bias
        for (int i = 0; i < NN->layers[l]->numNeuron; i++)
        {
            NN->layers[l]->bias->data[i][0] -= (0.01 * (layerErrors[l]->data[i][0]));
        }

        deleteMatrix(prevActT);
        deleteMatrix(deltas);
    }

    for (int l = 0; l < NN->numHiddenLayer; l++)
    {
        deleteMatrix(layerErrors[l]);
    }
    free(layerErrors);
}