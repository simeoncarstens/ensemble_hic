#define NO_IMPORT_ARRAY

#include "ensemble_hic.h"

static void set_neighbors(PyNBListObject *self) {
  /*
   * Generates relative grid indices of neighbors in a cubic grid.
   * Assumes that MAX_NO_NEIGHBORS is 13.
   */

  int j, k, n=self->n_cells, counter=1;

  /* i = 0 */  
  self->neighbors[0] = INDEX(0, 0, 1, n);

  for (k = -1; k < 2; k++, counter++) {
    self->neighbors[counter] = INDEX(0, 1, k, n);
  }

  /* i = 1 */
  for (j = -1; j < 2; j++) for (k = -1; k < 2; k++) {
    self->neighbors[counter] = INDEX(1, j, k, n);
    counter++;
  }
}

static int init_cells(PyNBListObject *self, int n_atoms) {

  int n_cells, i;

  n_cells = self->n_cells + 2;
  n_cells = n_cells * n_cells * n_cells;

  /* at most #atoms-cells can be non-empty */

  if (!(self->filled = MALLOC(n_atoms, Cell))) {
    RAISE(PyExc_StandardError, "init_cells: malloc failed.", -1);
  }

  for (i = 0; i < n_atoms; i++) {
    if (!(self->filled[i].objects = MALLOC(self->n_per_cell, int))) {
      RAISE(PyExc_StandardError, "init_cells: malloc failed.", -1);
    }
    self->filled[i].n_objects = 0;
  }

  if (!(self->cells = MALLOC(n_cells, Cell*))) {
    RAISE(PyExc_StandardError, "init_cells: malloc failed.", -1);
  }

  /* initially, all cells are empty. */

  for (i = 0; i < n_cells; i++) {
    self->cells[i] = NULL;
  }
  return 0;
}

static void del_cells(PyNBListObject *self) {

  int i;

  if (self->filled) {
    for (i = 0; i < self->n_atoms; i++) {      
      if (self->filled[i].objects) {
	free(self->filled[i].objects);
      }
    }
    free(self->filled);
  }
  if (self->cells) {
    for (i = 0; i < (self->n_cells+2) * (self->n_cells+2) * (self->n_cells+2) ; i++) {
      if (self->cells[i]) {
	free(self->cells[i]);
      }
    }
    free(self->cells);
  }
  self->filled   = NULL;
  self->cells    = NULL;
  self->n_filled = 0;
}

static double update_bbox(PyNBListObject *self, double *coords, int n) {
  /*
    Determine box that contains all coordinates. The box is stored in a
    3D vector: [x_min, x_max, x_max - x_min]
   */
  double x, x_min = coords[0], x_max = coords[0];
  int i;

  for (i=1; i < n; i++) {

    x = coords[i];
    if (x < x_min) {
      x_min = x;
    }
    else if (x > x_max) {
      x_max = x;
    }
  }

  self->origin = x_min - BOX_MARGIN;

  return x_max + BOX_MARGIN - x_min;
}

static int assign_atoms(PyNBListObject *self, vector *coords, int n_coords, int new_box) {
  /*
    Assigns the atoms to the cells of the grid that is used to 
    generate the non-bonded list.    
  */
  
  int i, j, k, n, index, n_filled=0;
  Cell *current_cell;

  double cellsize = self->cellsize;
  int *n_contacts = self->n_contacts;  
  int n_cells     = self->n_cells;

  if (!n_contacts) {
    RAISE(PyExc_StandardError, "assign_atoms: n_contacts has not been allocated.", -1);
  }

  /* determine new bounding box: if the bounding box cannot be 
     covered with 'n_cells' cells of size 'cellsize', we increase 
     the cell size until everthing fits. */

  if (new_box) {
    double size = update_bbox(self, (double*) coords, 3 * n_coords);
    if ((int) floor(size / cellsize) >= n_cells) {
      cellsize = size / (n_cells - 1. + 1.e-8);
    }
  }

  Cell **cells = self->cells;

  for (n=0; n < n_coords; n++) {

    /* set no. of interactions for all objects to 0 */
    
    n_contacts[n] = 0;

    /* project coordinates onto grid */
    
    i = (int) floor((coords[n][0] - self->origin) / cellsize);
    j = (int) floor((coords[n][1] - self->origin) / cellsize);
    k = (int) floor((coords[n][2] - self->origin) / cellsize);

    index = INDEX(i+1, j+1, k+1, n_cells);

    if (!cells[index]) {
      cells[index] = &self->filled[n_filled];
      cells[index]->id = index;
      n_filled++;
    }

    current_cell = cells[index];

    /* if cell is not empty, check if it has enough space */

    if (current_cell->n_objects >= self->n_per_cell) {
      printf("WARNING: no space left in cell (%d).\n", current_cell->id);
      continue;
    }

    /* add object to current cell */

    current_cell->objects[current_cell->n_objects] = n;
    current_cell->n_objects++;
  }
  self->n_filled = n_filled;

  return 0;
}

int nblist_update(PyNBListObject *self, vector *coords, int n_coords, int new_box) {
  /*
    The NB-list is generated as follows:
  */	     
 
  int i, j, k, n, partner_id, current_cell_id, total_n_contacts=0;
  int atom_id, *atom_contacts, *objects, *neighbor_objects;
  double sq_distance;
  Cell *current_cell, *neighbor;
  vector dx;
  
  if (!self->enabled) return -1;

  int *n_contacts       = self->n_contacts;
  int *neighbors        = self->neighbors;
  int **contacts        = self->contacts;
  double **sq_distances = self->sq_distances;
  double cellsize2      = self->cellsize * self->cellsize;

  // assign atoms to grid cells 

  if (assign_atoms(self, coords, n_coords, new_box)) return -1;

  // need to set points AFTER calling assign_atoms because
  // pointer might be allocated in this routine for the first time

  Cell **cells = self->cells;

  for (n=0; n < self->n_filled; n++) {

    current_cell    = &self->filled[n];
    objects         = current_cell->objects;
    current_cell_id = current_cell->id;

    /* loop through objects in current cell. */

    for (i=0; i < current_cell->n_objects; i++) {

      /* index of object 'i' */
      
      atom_id = objects[i];

      /* get interaction list for 'atom' and
	 set interaction counter for first atom. */

      atom_contacts = contacts[atom_id];

      /* intra-cell interactions */     

      for (j = i+1; j < current_cell->n_objects; j++) {

	/* get index of other interacting atom. */

	partner_id = objects[j];

	/* check if distance is larger than cell size */

	vector_sub(dx, coords[atom_id], coords[partner_id]);
	
	sq_distance = vector_dot(dx, dx);
	if (sq_distance > cellsize2) continue;

	/* add interaction i-j */

	atom_contacts[n_contacts[atom_id]] = partner_id;
	sq_distances[atom_id][n_contacts[atom_id]] = sq_distance;

	n_contacts[atom_id]++;

	total_n_contacts++;
      }
  
      /* inter-cell interactions */

      for (j=0; j < MAX_NO_NEIGHBORS; j++) {

        neighbor = cells[current_cell_id + neighbors[j]];

	/* if neighbor is empty, continue */

	if (!neighbor) {
	  continue;
	}

	neighbor_objects = neighbor->objects;
 
        /* loop through all objects in neighboring cell */

        for (k=0; k < neighbor->n_objects; k++) {

	  /* get index of other interacting atom. */

	  partner_id = neighbor_objects[k];

	  /* check if distance is larger than cell size */
	  
	  vector_sub(dx, coords[atom_id], coords[partner_id]);

	  sq_distance = vector_dot(dx, dx);
	  if (sq_distance > cellsize2) continue;

	  /* add interaction i-k */

	  atom_contacts[n_contacts[atom_id]] = partner_id;
	  sq_distances[atom_id][n_contacts[atom_id]] = sq_distance;

	  n_contacts[atom_id]++;

	  total_n_contacts++;
        }
      }
    } 
  }

  /* cleanup */

  for (i = 0; i < self->n_filled; i++) {
    self->filled[i].n_objects = 0;
    self->cells[self->filled[i].id] = NULL;
  }

  return total_n_contacts;
}

static PyObject *py_update(PyNBListObject *self, PyObject *args) {

  PyArrayObject *coords;
  int new_box;
  
  if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &coords, &new_box)) {
    return NULL;
  }

  int counter = nblist_update(self, (vector*) coords->data, coords->dimensions[0], new_box);

  return Py_BuildValue("i", counter);
}

static PyObject *py_update_bbox(PyNBListObject *self, PyObject *args) {

  PyArrayObject *coords;
  
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &coords)) {
    return NULL;
  }
  double size = update_bbox(self, (double*) coords->data, 3 * coords->dimensions[0]);

  return Py_BuildValue("d", size);
}

static PyMethodDef nblist_methods[] = {
  {"update", (PyCFunction) py_update, 1},
  {"update_bbox", (PyCFunction) py_update_bbox, 1},
  {NULL, NULL }
};

static void del_contacts(PyNBListObject *self) {

  int i;

  if (self->n_contacts) {
    free(self->n_contacts);
    self->n_contacts = NULL;
  }
  
  if (self->contacts) {
    for(i = 0; i < self->n_atoms; i++) {
      if (self->contacts[i]) {
	free(self->contacts[i]);
	self->contacts[i] = NULL;
      }
    }
    free(self->contacts);
    self->contacts = NULL;
  }

  if (self->sq_distances) {
    for(i = 0; i < self->n_atoms; i++) {
      if (self->sq_distances[i]) {
	free(self->sq_distances[i]);
	self->sq_distances[i] = NULL;
      }
    }
    free(self->sq_distances);
    self->sq_distances = NULL;
  }
}

static int set_natoms(PyNBListObject *self, int n) {

  int i, max_n;

  // estimate maximum number of interactions

  max_n = self->n_per_cell * (MAX_NO_NEIGHBORS+1);
  max_n = (max_n > n) ? n : max_n;
  
  self->max_n_contacts = max_n;

  /* free old interaction lists */

  del_contacts(self);

  /* allocate new interaction lists */
  
  if (!(self->n_contacts = MALLOC(n, int))) {
    RAISE(PyExc_StandardError, "set_natoms: malloc failed (n_contacts)", -1);
  }
  if (!(self->contacts = MALLOC(n, int*))) {
    RAISE(PyExc_StandardError, "set_natoms: malloc failed (contacts)", -1);
  }
  if (!(self->sq_distances = MALLOC(n, double*))) {
    RAISE(PyExc_StandardError, "set_natoms: malloc failed (sq_distances)", -1);
  }
    
  for(i=0; i < n; i++) {

    self->n_contacts[i]   = 0;
    self->contacts[i]     = NULL;
    self->sq_distances[i] = NULL;
    
    if (!(self->contacts[i] = MALLOC(max_n, int))) {
      RAISE(PyExc_StandardError, "set_natoms: malloc failed (contacts[i])", -1);
    }
    if (!(self->sq_distances[i] = MALLOC(max_n, double))) {
      RAISE(PyExc_StandardError, "set_natoms: malloc failed (sq_distances[i])", -1);
    }
  }
  self->n_atoms = n;

  return 0;
}

static void dealloc(PyNBListObject *self) {
  del_contacts(self);
  del_cells(self);
  PyObject_Del(self);
}

static PyObject *getattr(PyNBListObject *self, char *name) {

  int i, j, n, n_dims, max, dims[2];
  int *dummy;
  double *ddummy;

  if (!strcmp(name, "contacts")) {
    if (!self->contacts) {
      RETURN_PY_NONE;
    }
    n_dims  = 2;
    dims[0] = self->n_atoms;

    /* find maximum number of interactions */
    
    max = 0;
    for (i = 0; i < dims[0]; i++) {
      n = self->n_contacts[i];
      max = n > max ? n : max;
    }    

    dims[1] = max;

    /* create and fill dummy array */
    
    if (!(dummy = MALLOC(dims[0] * max, int))) {
      RAISE(PyExc_StandardError, "contacts: MALLOC failed", NULL);
    }
    
    for (i = 0; i < dims[0]; i++) {

      n = self->n_contacts[i];

      for (j = 0; j < n; j++) {
	dummy[i*max + j] = self->contacts[i][j];
      }
      for (; j < max; j++) {
	dummy[i*max + j] = -1;
      }
    }
    return PyArray_CopyFromDimsAndData(n_dims, dims, PyArray_INT, (char*) dummy);
  }

  else if (!strcmp(name, "sq_distances")) {

    if (!self->sq_distances) RETURN_PY_NONE;

    n_dims = 2;

    dims[0] = self->n_atoms;

    /* find maximum number of interactions */
    
    max = 0;
    for (i = 0; i < dims[0]; i++) {
      n = self->n_contacts[i];
      max = n > max ? n : max;
    }    

    dims[1] = max;

    /* create and fill dummy array */
    
    if (!(ddummy = MALLOC(dims[0] * max, double))) {
      RAISE(PyExc_StandardError, "sq_distances: MALLOC failed", NULL);
    }
    
    for (i = 0; i < dims[0]; i++) {

      n = self->n_contacts[i];

      for (j = 0; j < n; j++) {
	ddummy[i*max + j] = self->sq_distances[i][j];
      }
      for (; j < max; j++) {
	ddummy[i*max + j] = -1;
      }
    }

    return PyArray_CopyFromDimsAndData(n_dims, dims, PyArray_DOUBLE,
				       (char*) ddummy);
  }

  else if (!strcmp(name, "n_contacts")) {

    if (!self->n_contacts) RETURN_PY_NONE;

    n_dims = 1;

    dims[0] = self->n_atoms;

    return PyArray_CopyFromDimsAndData(n_dims, dims, PyArray_INT,
				       (char*) self->n_contacts);
  }

  else if (!strcmp(name, "neighbors")) {

    n_dims = 1;

    dims[0] = MAX_NO_NEIGHBORS;

    return PyArray_CopyFromDimsAndData(n_dims, dims, PyArray_INT,
				       (char*) self->neighbors);
  }

  else if (!strcmp(name, "enabled")) {
    return Py_BuildValue("i", self->enabled);
  }
  else if (!strcmp(name, "n_atoms")) {
    return Py_BuildValue("i", self->n_atoms);
  }
  else if (!strcmp(name, "n_filled")) {
    return Py_BuildValue("i", self->n_filled);
  }
  else if (!strcmp(name, "n_cells")) {
    return Py_BuildValue("i", self->n_cells);
  }
  else if (!strcmp(name, "cellsize")) {
    return Py_BuildValue("d", self->cellsize);
  }
  else if (!strcmp(name, "origin")) {
    return Py_BuildValue("d", self->origin);
  }
  else if (!strcmp(name, "n_per_cell")) {
    return Py_BuildValue("i", self->n_per_cell);
  }
  else {
    return Py_FindMethod(nblist_methods, (PyObject *)self, name);
  }
}

static int setattr(PyNBListObject *self, char *name, PyObject *op) {  

  if (!strcmp(name, "enabled")) {
    self->enabled = (int) PyInt_AsLong(op);
  }
  else if (!strcmp(name, "cellsize")) {
    self->cellsize = (double) PyFloat_AsDouble(op);
  }
  else if (!strcmp(name, "n_cells")) {
    self->n_cells = (int) PyInt_AsLong(op);
    set_neighbors(self);
    del_cells(self);
    init_cells(self, self->n_atoms);
  }
  else if (!strcmp(name, "n_per_cell")) {
    self->n_per_cell = (int) PyInt_AsLong(op);
  }
  else if (!strcmp(name, "n_atoms")) {
    return set_natoms(self, (int) PyInt_AsLong(op));
  }
  else {
    RAISE(PyExc_AttributeError, "Attribute does not exist or cannot be set", -1);
  }
  return 0;
}

static char __doc__[] = "Non-bonded list object"; 

PyTypeObject PyNBList_Type = { 
	PyObject_HEAD_INIT(0)
	0,			             /*ob_size*/
	"NBList",             	             /*tp_name*/
	sizeof(PyNBListObject),              /*tp_basicsize*/
	0,			             /*tp_itemsize*/
	(destructor)dealloc,                 /*tp_dealloc*/
	(printfunc)NULL,	             /*tp_print*/
       	(getattrfunc)getattr,                /*tp_getattr*/
	(setattrfunc)setattr,                /*tp_setattr*/
	(cmpfunc)NULL,         	             /*tp_compare*/
	(reprfunc)NULL,	                     /*tp_repr*/

	NULL,		                     /*tp_as_number*/
	NULL,	                             /*tp_as_sequence*/
	NULL,		 	             /*tp_as_mapping*/

	(hashfunc)0,		             /*tp_hash*/
	(ternaryfunc)0,		             /*tp_call*/
	(reprfunc)0,		             /*tp_str*/
		
	0L,0L,0L,0L,
	__doc__                              /* Documentation string */
};

PyObject * PyNBList_nblist(PyObject *self, PyObject *args) {

  PyNBListObject *object;

  if (!PyArg_ParseTuple(args, "")) return NULL;

  object = PyObject_NEW(PyNBListObject, &PyNBList_Type);

  object->n_cells      = 0;
  object->cellsize     = 0.;
  object->n_per_cell   = 0;
  object->contacts     = NULL;
  object->sq_distances = NULL;
  object->n_contacts   = NULL;
  object->cells        = NULL;
  object->filled       = NULL;
  object->n_filled     = -1;

  object->max_n_contacts = -1;
  object->enabled = 0;
  
  return (PyObject *) object;
}


