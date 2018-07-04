#ifndef __ENSEMBLE_HIC_H__
#define __ENSEMBLE_HIC_H__

#ifdef __cplusplus
 extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nblist.h"

#define RAISE(a,b,c) {PyErr_SetString(a, b); return c;}
#define HERE {printf("%s: %d\n",__FILE__,__LINE__);}
#define MALLOC(n, t) ((t*) malloc((n) * sizeof(t)))
#define RETURN_PY_NONE {Py_INCREF(Py_None);return Py_None;}


#ifdef __cplusplus
 }
#endif

#endif
