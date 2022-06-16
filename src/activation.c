#include <stdio.h>
#include "../src/matrix.h"
#include "../src/activation.h"

Matrix* softMax(Matrix* m)
{
    Matrix* result = copyMatrix(m);
    matrixApply(result, &toExp);
    double factor = 0;
    for (int r = 0; r < result->row; r++)
    {
        for (int c = 0; c < result->col; c++)
        {
            factor += result->data[r][c];
        }
    }
    factor = 1 / factor;

    matrixScale(result, factor);
    return result;
}

Matrix* dSoftMax(Matrix* m)
{
    Matrix* result = copyMatrix(m);
    for (int r = 0; r < result->row; r++)
    {
        for (int c = 0; c < result->col; c++)
        {
            result->data[r][c] = m->data[r][c] * (1 - m->data[r][c]);
        }
    }
    return result;
}

Matrix* argMax(Matrix* m)
{
    if (m->col != 1)
    {
        printDimensionError();
        return NULL;
    }

    double max = m->data[0][0];
    int index = 0;
    for(int i = 1; i < m->row; i++)
    {
        if (m->data[i][0] > max)
        {
            max = m->data[i][0];
            index = i;
        }
    }
    Matrix* result = makeMatrix(m->row, 1);
    result->data[index][0] = 1;
    return result;
}