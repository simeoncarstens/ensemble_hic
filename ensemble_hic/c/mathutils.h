#ifndef MATHUTILS_H
#define MATHUTILS_H

#include "ensemble_hic.h"

typedef double vector_double[3];

#define vector vector_double

#define array vector

PyObject *PyArray_CopyFromDimsAndData(int n_dimensions, int *dimensions, 
				      int type_num, char *data);

void   vector_print(vector x);
double vector_dot(vector a, vector b);
void   vector_sub(vector dest, vector a, vector b);

#endif
