#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image.h"

#define TRAINING_PATH "./data/mnist_train.csv"
#define TESTING_PATH "./data/mnist_test.csv"
#define MAX_STRING_LENGTH 8192

Image** loadImages(int num, char test)
{
    FILE* file;
    if (test)
    {
        printf("Load %d images for testing\n", num);
        file = fopen(TESTING_PATH, "r");
    }
    else
    {
        printf("Load %d images for training\n", num);
        file = fopen(TRAINING_PATH, "r");
    }

    if (file == NULL)
    {
        printf("Error loading file %s\n", TRAINING_PATH);
        return NULL;
    }

    Image** result = malloc(sizeof(Image*) * num);
    char buff[MAX_STRING_LENGTH];
    fgets(buff, MAX_STRING_LENGTH, file);
    for (int i = 0; i < num; i++)
    {
        result[i] = malloc(sizeof(Image));
        result[i]->data = malloc(sizeof(double) * 784);
        result[i]->label = calloc(10, sizeof(double));
        result[i]->dataLen = 784;
        result[i]->labelLen = 10;

        fgets(buff, MAX_STRING_LENGTH, file);

        char* tok = strtok(buff, ",");
        int countInLine = 0;
        while (tok != NULL)
        {
            if (countInLine == 0)
            {
                result[i]->label[atoi(tok)] = 1;
                result[i]->labelValue = atoi(tok);
            }
            else
            {
                result[i]->data[countInLine-1] = atoi(tok) / (double)256;
            }
            tok = strtok(NULL, ","); // Pass NULL to continue reading
            countInLine++;
        }
    }
    
    fclose(file);
    return result;
}

void printImage(Image* img)
{
    for (int i = 0; i < img->dataLen; i++)
    {
        if (i % img->dataLen == 0 && i != 0)
        {
            printf("\n");
        }
        printf("%5.2lf ", img->data[i]);
    }
}

void deleteImage(Image* img)
{
    free(img->data);
    free(img->label);
    free(img);
    img = NULL;
}

void deleteImages(Image** imgs, int num)
{
    for (int i = 0; i < num; i++)
    {
        deleteImage(imgs[i]);
    }
    free(imgs);
    imgs = NULL;
}