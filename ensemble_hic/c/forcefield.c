/*
Copyright by Michael Habeck (2016)
*/
#define NO_IMPORT_ARRAY

#include "ensemble_hic.h"

double forcefield_energy(PyForceFieldObject *self, 
			 double *coordinates, 
			 int *types,
			 int n_particles) {
  /*
    Evaluates non-bonded interactions based on the current neighbor list. 
   */
  if (!self->enabled) return 0.;

  int n_types     = self->n_types;
  int *n_contacts = self->nblist->n_contacts;
  int **contacts  = self->nblist->contacts;
  vector *coords  = (vector*) coordinates;

  double *k = self->k;
  double *d = self->d;

  double E=0., r;
  int    *contacts_i, index, i, j, type_i, n;
  vector dx;
  
  /* loop through interactions of all atoms */

  for (i = 0; i < n_particles; i++) {

    type_i     = types[i] * n_types;
    contacts_i = contacts[i];

    for (n = 0; n < n_contacts[i]; n++) {

      j = contacts_i[n];
      index = type_i + types[j];

      vector_sub(dx, coords[i], coords[j]);
      r = sqrt(vector_dot(dx, dx));

      E += self->f((PyObject*) self, r, d[index], k[index]);
    }
  }

  /* non-bonded overall force constant */

  E *= self->K;

  return E;
}

double forcefield_gradient(PyForceFieldObject *self, 
			   double *coordinates,
			   double *gradient,
			   int *types, 
			   int n_particles,
			   double *E_ptr) {
  /*
    Evaluates the non-bonded energy and its gradient based on the current neighbor list. 
   */
  if (!self->enabled) return -1;

  int n_types     = self->n_types;
  int *n_contacts = self->nblist->n_contacts;
  int **contacts  = self->nblist->contacts;

  double *k = self->k;
  double *d = self->d;

  vector *forces = (vector*) gradient;
  vector *coords = (vector*) coordinates;

  double E=0., r, c;
  int    *contacts_i, index, i, j, n, l, type_i;
  vector dx;

  for (i = 0; i < n_particles; i++) {

    /* for the first atom, get atom-type */

    type_i     = types[i] * n_types;
    contacts_i = contacts[i];

    /* loop through all interaction partners of atom 'atom' */

    for (n = 0; n < n_contacts[i]; n++) {
      
      j = contacts_i[n];
      index = type_i + types[j];
      
      vector_sub(dx, coords[i], coords[j]);
      r = sqrt(vector_dot(dx, dx));
 
      c = self->grad_f((PyObject*) self, r, d[index], k[index], &E);

      for (l = 0; l < 3; l++) {
	forces[i][l] += c * dx[l];
	forces[j][l] -= c * dx[l];
      }
    }
  }
  if (E_ptr) {
    *E_ptr = E;
  }
  return 0;
}

int forcefield_set_k(PyForceFieldObject *self, PyObject *op){
  
  int n_types, i, j, s0, s1;
  double *k;

  PyArrayObject *K;

  if (!PyArray_Check(op)) {
    RAISE(PyExc_TypeError, "numpy array expected (set_k).", -1);
  }
  K = (PyArrayObject*) op;
  
  n_types = self->n_types;

  if (K->nd != 2) {
    RAISE(PyExc_ValueError, "Array must be of rank 2.", -1);
  }
  if ((K->dimensions[0] != n_types) || (n_types != K->dimensions[1])) {
    RAISE(PyExc_ValueError, "rank must be (n_types, n_types).", -1);
  }
  if (!(k = MALLOC(n_types * n_types, double))) {
    RAISE(PyExc_StandardError, "malloc failed (set_k)", -1);
  }

  s0 = K->strides[0];
  s1 = K->strides[1];

  for (i = 0; i < n_types; i++) for (j = 0; j < n_types; j++) {
    k[i * n_types + j] = * (double*) (K->data + i * s0 + j * s1);
    }
  if (self->k) {
    free(self->k);
  }

  self->k = k;

  return 0;
}

int forcefield_set_d(PyForceFieldObject *self, PyObject *op){
  
  int n_types, i, j, s0, s1;
  double *d;

  PyArrayObject *D;

  if (!PyArray_Check(op)) {
    RAISE(PyExc_TypeError, "numpy array expected (set_d).", -1);
  }
  D = (PyArrayObject*) op;
  
  n_types = self->n_types;

  if (D->nd != 2) {
    RAISE(PyExc_ValueError, "Array must be of rank 2.", -1);
  }
  if ((D->dimensions[0] != n_types) || (n_types != D->dimensions[1])) {
    RAISE(PyExc_ValueError, "rank must be (n_types, n_types).", -1);
  }
  if (!(d = MALLOC(n_types * n_types, double))) {
    RAISE(PyExc_StandardError, "malloc failed (set_d)", -1);
  }
  s0 = D->strides[0];
  s1 = D->strides[1];

  for (i = 0; i < n_types; i++) for (j = 0; j < n_types; j++) {
    d[i * n_types + j] = * (double*) (D->data + i * s0 + j * s1);
    }

  if (self->d) {
    free(self->d);
  }
  self->d = d;

  return 0;
}

void forcefield_dealloc(PyForceFieldObject *self) {

  if (self->k) {
    free(self->k); 
  }
  if (self->d) {
    free(self->d); 
  }
  if (self->nblist) {
    Py_DECREF((PyObject*) self->nblist);
  }
  self->k = NULL;
  self->d = NULL;

  self->nblist = NULL;
}

void forcefield_init(PyForceFieldObject *self) {

  self->n_types = 0;
  self->enabled = 1;
  self->K       = 1.;

  self->k = NULL;
  self->d = NULL;

  self->nblist = NULL;
}

PyObject * forcefield_getattr(PyForceFieldObject *self, char *name) {

  int n_dims;
  int dims[2];
  
  if (!strcmp(name, "k")) {
    
    if (self->k) {
      n_dims = 2;
      dims[0] = dims[1] = self->n_types;
      
      return PyArray_CopyFromDimsAndData(n_dims, dims, PyArray_DOUBLE,
					 (char *) self->k);
    }
    else RETURN_PY_NONE
  }
  else if (!strcmp(name, "d")) {
    
    if (self->d) {
      n_dims = 2;
      dims[0] = dims[1] = self->n_types;

      return PyArray_CopyFromDimsAndData(n_dims, dims, PyArray_DOUBLE,
					 (char *) self->d);
    }
    else RETURN_PY_NONE;
  }
  else if (!strcmp(name, "nblist")) {

    if (self->nblist) {
      Py_INCREF((PyObject*) self->nblist);
      return (PyObject*) self->nblist;
    }
    else RETURN_PY_NONE;
  }
  else if (!strcmp(name, "n_types")) {
    return Py_BuildValue("i", self->n_types);
  }
  else if (!strcmp(name, "enabled")) {
    return Py_BuildValue("i", self->enabled);
  }
  else if (!strcmp(name, "K")) {
    return Py_BuildValue("d", self->K);
  }
  return NULL;
}

int forcefield_setattr(PyForceFieldObject *self, char *name, PyObject *op) {

  int n_types;

  if (!strcmp(name, "enabled")) {
    self->enabled = (int) PyInt_AsLong(op);
  }
  else if (!strcmp(name, "n_types")) {
    n_types = (int) PyInt_AsLong(op);
    
    if (n_types != self->n_types) {
      if (self->k) free(self->k);
      if (self->d) free(self->d);
      self->k = NULL;
      self->d = NULL;
      self->n_types = n_types;
    }
  }
  else if (!strcmp(name, "K")) {
    self->K = (double) PyFloat_AsDouble(op);
  }
  else if (!strcmp(name, "k")) {
    forcefield_set_k(self, op);
  }
  else if (!strcmp(name, "d")) {
    forcefield_set_d(self, op);
  }
  else if (!strcmp(name, "nblist")) {

    if (self->nblist) Py_DECREF((PyObject*) self->nblist);
    
    self->nblist = (PyNBListObject*) op;
    Py_INCREF(op);
  }
  else {
    return 0;
  }
  return 1;
}

