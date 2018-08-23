import numpy

def kth_diag_indices(a, k):
    """
    From
    https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    """
    rows, cols = numpy.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def load_pdb(filename):

    from csb.bio.io.wwpdb import StructureParser

    return StructureParser(filename).parse_structure().get_coordinates()
