#include <stdio.h>
#include "../src/matrix.h"
#include "../src/neural.h"
#include "../src/image.h"
#include "../src/test.h"

int main(int argc, char *argv[])
{
    // [num input nodes] [num hidden layers] [num hidden layers nodes...]
    int config[] = {784, 3, 30, 20, 10};

    NeuralNetwork* test = makeNeuralNetwork(config, 5);

    Image** images = loadImages(60000, 0);
    for (int i = 0; i < 60000; i++)
    {
        if (i % 2500 == 0)
        {
            printf("Training... (%d/60000)\n", i);
        }
        backPropogate(test, images[i]->data, images[i]->label);
    }

    deleteImages(images, 60000);

    testAll(test);

    deleteNeuralNetwork(test);

    return 0;
}