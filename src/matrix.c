#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/matrix.h"

Matrix* makeMatrix(int row, int col)
{
    Matrix* result = malloc(sizeof(Matrix));
    result->row = row;
    result->col = col;

    result->data = malloc(sizeof(double*) * row);
    for (int i = 0; i < row; i++)
    {
        result->data[i] = calloc(col, sizeof(double));
    }
    return result;
}

Matrix* copyMatrix(Matrix* m)
{
    Matrix* result = makeMatrix(m->row, m->col);
    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            result->data[r][c] = m->data[r][c];
        }
    }
    return result;
}

void deleteMatrix(Matrix* m)
{
    for (int r = 0; r < m->row; r++)
    {
        free(m->data[r]);
    }
    free(m->data);
    free(m);
    m = NULL;
}

void fillMatrix(Matrix* m, double x)
{
    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            m->data[r][c] = x;
        }
    }
}

Matrix* matrixAdd(Matrix* m1, Matrix* m2)
{
    if (m1->row != m2->row || m1->col != m2->col)
    {
        printDimensionError();
        return NULL;
    }

    int row = m1->row;
    int col = m1->col;
    Matrix* result = makeMatrix(row, col); 
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < col; c++)
        {
            result->data[r][c] = m1->data[r][c] + m2->data[r][c];
        }
    }
    return result;
}

Matrix* matrixSub(Matrix* m1, Matrix* m2)
{
    if (m1->row != m2->row || m1->col != m2->col)
    {
        printDimensionError();
        return NULL;
    }

    int row = m1->row;
    int col = m1->col;
    Matrix* result = makeMatrix(row, col); 
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < col; c++)
        {
            result->data[r][c] = m1->data[r][c] - m2->data[r][c];
        }
    }
    return result;
}

Matrix* matrixMul(Matrix* left, Matrix* right)
{
    if (right->row != left->col)
    {
        printDimensionError();
        return NULL;
    }

    Matrix* product = makeMatrix(left->row, right->col);
    for (int r = 0; r < product->row; r++)
    {
        for (int c = 0; c < product->col; c++)
        {
            double sum = 0;
            for (int n = 0; n < right->row; n++)
            {
                sum += left->data[r][n] * right->data[n][c];
            }
            product->data[r][c] = sum;
        }
    }

    return product;
}

Matrix* matrixFlatten(Matrix* m, int order, int axis)
{
    if (!(order == 0 || order == 1) || !(axis == 0 || axis == 1))
    {
        printDimensionError();
        return NULL;
    }

    if (axis == 0)
    {
        Matrix* result = makeMatrix(1, m->row * m->col);
        for (int r = 0; r < m->row; r++)
        {
            for (int c = 0; c < m->col; c++)
            {
                if (order == 0)
                {
                    result->data[0][r*m->col + c] = m->data[r][c];
                }
                else
                {
                    result->data[0][c*m->row + r] = m->data[r][c];
                }
            }
        }
        return result;
    }
    else
    {
        Matrix* result = makeMatrix(m->row * m->col, 1);
        for (int r = 0; r < m->row; r++)
        {
            for (int c = 0; c < m->col; c++)
            {
                if (order == 0)
                {
                    result->data[r*m->col + c][0] = m->data[r][c];
                }
                else
                {
                    result->data[c*m->row + r][0] = m->data[r][c];
                }
            }
        }
        return result;
    }
}

Matrix* matrixHadaMul(Matrix* m1, Matrix* m2)
{
    if ((m1->col != m2->col) || (m1->row != m2->row))
    {
        printDimensionError();
        return NULL;
    }
    Matrix* result = makeMatrix(m1->row, m1->col);
    for (int r = 0; r < result->row; r++)
    {
        for (int c = 0; c < result->col; c++)
        {
            result->data[r][c] = m1->data[r][c] * m2->data[r][c];
        }
    }
    return result;
}

void matrixScale(Matrix* m, double s)
{
    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            m->data[r][c] *= s;
        }
    }
}

void matrixAddScalar(Matrix* m, double s)
{
    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            m->data[r][c] += s;
        }
    }
}

void matrixApply(Matrix* m, double (*func)(double))
{
    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            m->data[r][c] = (*func)(m->data[r][c]);
        }
    }
}

void matrixRandomize(Matrix* m)
{
    time_t t;
    srand((unsigned) time(&t));

    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            m->data[r][c] = ((rand() % 101) - 50) / (double) 100;
        }
    }
}

Matrix* matrixTranspose(Matrix* m)
{
    Matrix* result = makeMatrix(m->col, m->row);
    for (int r = 0; r < result->row; r++)
    {
        for (int c = 0; c < result->col; c++)
        {
            result->data[r][c] = m->data[c][r];
        }
    }
    return result;
}

void printMatrix(Matrix* m)
{
    for (int r = 0; r < m->row; r++)
    {
        for (int c = 0; c < m->col; c++)
        {
            printf("%5.2lf ", m->data[r][c]);
        }
        printf("\n");
    }
}

void printDimensionError()
{
    printf("Dimensions do not match\n");
}