#define NO_IMPORT_ARRAY

#include "ensemble_hic.h"

void vector_print(vector x) {
  int i;
  
  printf("[ ");
  for (i=0;i<3;i++) printf("%e ",x[i]);
  printf("]\n");
}

double vector_dot(vector a, vector b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void vector_sub(vector dest, vector a, vector b) {
  
  int i;

  for (i = 0; i < 3; i++) {
    dest[i] = a[i] - b[i];
  }
}

PyObject *PyArray_CopyFromDimsAndData(int n_dimensions, int *dimensions, 
				      int type_num, char *data) {
  /*
    This method is similar to PyArray_FromDimAndData. It creates a new
    PyArrayObject, but instead of referencing 'data', it returns a
    copy of it.
   */

  PyObject *a1, *a2;

  npy_intp dims[n_dimensions];

  int i;
  for (i = 0; i < n_dimensions; i++) {
      dims[i] = dimensions[i];
  }
  a1 = PyArray_SimpleNewFromData(n_dimensions, dims, type_num, data);
  a2 = PyArray_Copy((PyArrayObject*) a1);

  Py_DECREF(a1);

  return PyArray_Return((PyArrayObject*) a2);
}

