/*
Copyright by Michael Habeck (2016)
*/
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
#include "mathutils.h"
   
#define RAISE(a,b,c) {PyErr_SetString(a, b); return c;}
#define HERE {printf("%s: %d\n",__FILE__,__LINE__);}
#define MALLOC(n, t) ((t*) malloc((n) * sizeof(t)))
#define RETURN_PY_NONE {Py_INCREF(Py_None);return Py_None;}

typedef double (*forcefield_energyfunc) (PyObject*, PyObject*);
typedef int (*forcefield_gradientfunc) (PyObject*, PyObject*, double *);

typedef double (*forcefield_energyterm) (PyObject*, double, double, double);
typedef double (*forcefield_gradenergyterm) (PyObject*, double, double, double, double *);

#define PyForceFieldObject_HEAD \
        PyObject_HEAD \
        double K;                /* overall force constant */ \
	int enabled;             /* switch for turning ON / OFF the interaction */\ 
	int n_types;             /* number of atom-types supported by the force-field */\
        double *k;               /* matrix of pairwise force-constants */\
        double *d;               /* matrix of pairwise sums of vwd-radii */\
        PyNBListObject *nblist;  /* neighbor list */\
        forcefield_energyterm f; \
        forcefield_gradenergyterm grad_f; \
	forcefield_energyfunc   energy; \
        forcefield_gradientfunc gradient;

/* Force fields */

typedef struct {
  PyForceFieldObject_HEAD
} PyForceFieldObject;

typedef struct {
  PyForceFieldObject_HEAD
} PyProlsqObject;

extern PyTypeObject PyProlsq_Type;

// general force field routines

int        forcefield_set_k(PyForceFieldObject *self, PyObject *op);
int        forcefield_set_d(PyForceFieldObject *self, PyObject *op);
void       forcefield_init(PyForceFieldObject *self);
void       forcefield_dealloc(PyForceFieldObject *self);
int        forcefield_setattr(PyForceFieldObject *self, char *name, PyObject *op);
PyObject * forcefield_getattr(PyForceFieldObject *self, char *name);
double     forcefield_energy(PyForceFieldObject *self, double *coords, int *types, int n_particles);
double     forcefield_gradient(PyForceFieldObject *self, double *coords, double *forces, int *types, int n_particles, double *E);

PyObject * PyProlsq_New(PyObject *self, PyObject *args);

#ifdef __cplusplus
 }
#endif

#endif
