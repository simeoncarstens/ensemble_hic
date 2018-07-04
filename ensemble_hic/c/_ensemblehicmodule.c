#include "ensemble_hic.h"

#ifdef __cplusplus
 extern "C" {
#endif

static PyMethodDef methods[] = {
  {"nblist", (PyCFunction) PyNBList_nblist, 1},
  {NULL, NULL}
};


void init_ensemble_hic(void) {

  import_array();
  Py_InitModule("_ensemble_hic", methods);

  /* set object types correctly */

  PyNBList_Type.ob_type = &PyType_Type;
}

#ifdef __cplusplus
}
#endif
