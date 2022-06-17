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
    result->numHiddenLayer = config[1];
    result->layers = malloc(sizeof(Layer*) * result->numHiddenLayer);
    result->inputs = makeMatrix(result->numInput, 1);

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
    free(NN);
    NN = NULL;
}

void feedForward(NeuralNetwork* NN, double* inputs)
{
    // Copying array into input matrix
    for (int i = 0; i < NN->numInput; i++)
    {
        NN->inputs->data[i][0] = inputs[i];
    }

    // Move forward through layers
    for (int i = 0; i < NN->numHiddenLayer; i++)
    {
        deleteMatrix(NN->layers[i]->activations);
        deleteMatrix(NN->layers[i]->z);
        Matrix* zPreBias = matrixMul(NN->layers[i]->w, i == 0 ? NN->inputs : NN->layers[i-1]->activations);
        NN->layers[i]->z = matrixAdd(zPreBias, NN->layers[i]->bias);

        // Activation func for last layer is reserved for softMax, others use Leaky ReLU
        if (i != NN->numHiddenLayer-1)
        {
            NN->layers[i]->activations = copyMatrix(NN->layers[i]->z); // activation takes in z (weight input)
            matrixApply(NN->layers[i]->activations, &lrelu);
        }
        else
        {
            NN->layers[i]->activations = softMax(NN->layers[i]->z);
        }

        deleteMatrix(zPreBias);
    }
}

void predict(NeuralNetwork* NN, double* inputs)
{
    // Copying array into input matrix
    for (int i = 0; i < NN->numInput; i++)
    {
        NN->inputs->data[i][0] = inputs[i];
    }

    // Move forward through layers
    for (int i = 0; i < NN->numHiddenLayer; i++)
    {
        deleteMatrix(NN->layers[i]->activations);
        deleteMatrix(NN->layers[i]->z);
        Matrix* zPreBias = matrixMul(NN->layers[i]->w, i == 0 ? NN->inputs : NN->layers[i-1]->activations);
        NN->layers[i]->z = matrixAdd(zPreBias, NN->layers[i]->bias);

        // Activation func for last layer is reserved for argMax. others use Leaky ReLU
        if (i != NN->numHiddenLayer-1)
        {
            NN->layers[i]->activations = copyMatrix(NN->layers[i]->z);
            matrixApply(NN->layers[i]->activations, &lrelu);
        }
        else
        {
            NN->layers[i]->activations = argMax(NN->layers[i]->z);
        }

        deleteMatrix(zPreBias);
    }
}

void backPropogate(NeuralNetwork* NN, double* inputs, double* observedValues)
{
    // Updtate the output in NN
    feedForward(NN, inputs);

    // Copying array into matrix
    Matrix* observed = makeMatrix(NN->layers[NN->numHiddenLayer-1]->numNeuron, 1);
    for (int r = 0; r < NN->layers[NN->numHiddenLayer-1]->numNeuron; r++)
    {
        observed->data[r][0] = observedValues[r];
    }

    // Compute error of the output layer
    // Math detail: https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    Matrix** layerErrors = malloc(sizeof(Matrix*) * NN->numHiddenLayer);
    layerErrors[NN->numHiddenLayer-1] = matrixSub(NN->layers[NN->numHiddenLayer-1]->activations, observed); // Derivative of CE w.r.t the z of output layer
    deleteMatrix(observed);

    // Back propogation starting with the second last hidden layer and working backward
    // Every layer looks at its next layer's errors and calculates its own accordingly
    for (int l = NN->numHiddenLayer-2; l >= 0; l--)
    {
        Matrix* nextT = matrixTranspose(NN->layers[l+1]->w);
        Matrix* nextError = matrixMul(nextT, layerErrors[l+1]);
        Matrix* dLrelu = copyMatrix(NN->layers[l]->z);
        matrixApply(dLrelu, &dlrelu);
        layerErrors[l] = matrixHadaMul(nextError, dLrelu);

        deleteMatrix(nextT);
        deleteMatrix(nextError);
        deleteMatrix(dLrelu);
    }

    // Calculate gradients for weights and biases
    for (int l = 0; l < NN->numHiddenLayer; l++)
    {
        // Calculate matrix delta whose dimensions match current layer's w
        // It represents all gradients of the current layer's w
        // Math detail: http://neuralnetworksanddeeplearning.com/chap2.html
        Matrix* prevActT = matrixTranspose(l == 0 ? NN->inputs : NN->layers[l-1]->activations);
        Matrix* deltas = matrixMul(layerErrors[l], prevActT);
        
        // Update to current layer's weights
        for (int r = 0; r < deltas->row; r++)
        {
            for (int c = 0; c < deltas->col; c++)
            {
                NN->layers[l]->w->data[r][c] -= (0.01 * (deltas->data[r][c]));
            }
        }

        // Update to current layer's biases
        // Delta for biases happens to be the current error
        // Math detail: http://neuralnetworksanddeeplearning.com/chap2.html
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