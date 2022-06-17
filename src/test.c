#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/test.h"

void testAll(NeuralNetwork* NN)
{
    Image** images = loadImages(10000, 1);
    int correct = 0;

    for (int i = 0; i < 10000; i++)
    {
        predict(NN, images[i]->data);
        int predicted = -1;
        for (int j = 0; j < NN->layers[NN->numHiddenLayer-1]->numNeuron; j++)
        {
            if (NN->layers[NN->numHiddenLayer-1]->activations->data[j][0] == 1)
            {
                predicted = j;
            }
        }
        if (predicted == images[i]->labelValue)
        {
            correct++;
        }
    }
    printf("Accuracy: %5.2f%% (%d/10000)\n", 100 * correct/10000.0, correct);

    deleteImages(images, 10000);
}

void testRandom(NeuralNetwork* NN)
{
    Image** images = loadImages(10000, 1);

    time_t t;
    srand((unsigned) time(&t));

    int rndIndex = rand() % 10000;
    predict(NN, images[rndIndex]->data);
    int predicted = -1;
    for (int j = 0; j < NN->layers[NN->numHiddenLayer-1]->numNeuron; j++)
    {
        if (NN->layers[NN->numHiddenLayer-1]->activations->data[j][0] == 1)
        {
            predicted = j;
        }
    }

    printf("Predicts: %d\n", predicted);
    printf("Observes: %d\n", images[rndIndex]->labelValue);
    deleteImages(images, 10000);
}