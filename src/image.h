#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include "../src/matrix.h"

typedef struct
{
    double* data;
    double* label;
    int dataLen;
    int labelLen;
    int labelValue;
} Image;

Image** loadImages(int num, char test);
void printImage(Image* img);
void deleteImage(Image* img);
void deleteImages(Image** imgs, int num);

#endif