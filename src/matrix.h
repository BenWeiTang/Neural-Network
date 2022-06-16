#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    double** data;
    int row;
    int col;
} Matrix;

Matrix* makeMatrix(int row, int col);
Matrix* copyMatrix(Matrix* m);
void deleteMatrix(Matrix* m);
void fillMatrix(Matrix* m, double x);
Matrix* matrixAdd(Matrix* m1, Matrix* m2);
Matrix* matrixSub(Matrix* m1, Matrix* m2);
Matrix* matrixMul(Matrix* left, Matrix* right);
Matrix* matrixFlatten(Matrix* m, int order, int axis);
Matrix* matrixHadaMul(Matrix* m1, Matrix* m2);
void matrixScale(Matrix* m, double s);
void matrixAddScalar(Matrix* m, double s);
void matrixApply(Matrix* m, double (*func)(double));
void matrixRandomize(Matrix* m);
Matrix* matrixTranspose(Matrix* m);
void printMatrix(Matrix* m);
void printDimensionError();

#endif