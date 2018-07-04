#ifndef NBLIST_H
#define NBLIST_H

#define MAX_NO_NEIGHBORS  13     /* maximal number of neighbor cells in
				    a cubic grid */

#define INDEX(i, j, k, n) (n * (n * i + j) + k)

#define BOX_MARGIN 1e-5

typedef struct _Cell {

  int id;                  /* every cell has a unique id */
  int n_objects;           /* the no. of objects contained in the cell */
  int *objects;            /* the list of object ids */

} Cell;

typedef struct _PyNBListObject {

  PyObject_HEAD

  int *n_contacts;         /* array that stores the number of interaction
			      partners for each atom */

  int **contacts;

  double **sq_distances; 

  int n_cells;             /* the grid is assumed to be cubic with
			      'n_cells' cells in each direction */
  double cellsize;         /* the cells are assumed to be also cubic
			      with extend 'cellsize'. */
  int n_atoms;             /* no. of atoms (private variable), needed to
			      to allocate the interaction lists */
  int n_per_cell;          /* the max. no. of atoms that can be stored
			      in one grid-cell. */
  int n_filled;            /* number of cells that contain at least 
			      one atom */
  int max_n_contacts;      /* internal variable that stores the maximal
			      number of interaction partners for one
			      atom, this will be the 2nd dimension of
			      interactionList */
  Cell **cells;            /* pointers to a grid of cells (most of them 
			      will point to NULL) */
  Cell *filled;            /* list of the non-empty cells for some state */

  int enabled;

  double origin;           // origin of bounding box
  
  int neighbors[MAX_NO_NEIGHBORS];

} PyNBListObject;

extern PyTypeObject PyNBList_Type;

PyObject * PyNBList_nblist(PyObject *self, PyObject *args);

#endif
