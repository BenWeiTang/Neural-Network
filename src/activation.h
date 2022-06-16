#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

static inline double sigmoid(double x) {return 1 / (1 + exp(x)); }
static inline double dSidgmoid(double x) { return x * (1 - x); }
static inline double relu(double x) { return x > 0 ? x : 0; }
static inline double drelu(double x) { return x > 0 ? 1 : 0; }
static inline double lrelu(double x) { return x > 0 ? x : 0.05 * x; }
static inline double dlrelu(double x) { return x > 0 ? 1 : 0.05; }
static inline double toExp(double x) { return exp(x); }

Matrix* softMax(Matrix* m);
Matrix* dSoftMax(Matrix* m);
Matrix* argMax(Matrix* m);

#endif