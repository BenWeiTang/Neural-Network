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
        factor += result->data[r][0];
    }
    factor = 1 / factor;

    matrixScale(result, factor);
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