"""
Script to parse 5C data from Nora et al., Nature 2012
"""

import os
import numpy as np

def parse_5C_file(filename):
    """
    Reads the raw 5C data file and returns reverse restriction fragments,
    forward restriction fragments, and a matrix of shape
    (# forward fragments + 2, # reverse fragments + 2).
    First two rows are start / end genomic coordinates of reverse restriction
    fragments, first two columns are start / end genomic coordinates of forward
    restriction fragments.
    Genomic coordinates are hg11
    """
    data = open(filename).readlines()
    data = data[7:]
    data = [y.split('\t') for y in data]
    data = np.array(data)

    rev_fragments = [x[x.find('chrX:')+5:] for x in data[0]]
    rev_fragments = [x.split('-') for x in rev_fragments]
    rev_fragments = [(int(x[0]), int(x[1])) for x in rev_fragments[1:]]
    rev_fragments = np.array(rev_fragments).swapaxes(1,0)

    for_fragments = [x[x.find('chrX:')+5:] for x in data[1:,0]]
    for_fragments = [x.split('-') for x in for_fragments]
    for_fragments = [(int(x[0]), int(x[1])) for x in for_fragments]
    for_fragments = np.array(for_fragments)

    matrix = np.zeros((len(for_fragments) + 2, len(rev_fragments.T) + 2))
    matrix[2:,:2] = for_fragments
    matrix[:2,2:] = rev_fragments
    matrix[2:,2:] = data[1:,1:]

    return rev_fragments, for_fragments, matrix

def extract_region(matrix, region_start, region_end):
    """
    Extracts a region from a matrix produced by parse_5C_file.
    Returns the reverse and forward restriction fragments in the region
    and the part of the matrix covered by the region
    """
    land = np.logical_and
    region_row_mask = land(matrix[:,0] >= region_start, matrix[:,1] <= region_end)
    region_col_mask = land(matrix[0,:] >= region_start, matrix[1,:] <= region_end)
    region = matrix[region_row_mask]
    region = region[:,region_col_mask]

    region_fors = matrix[region_row_mask, :2]
    region_revs = matrix[:2, region_col_mask]
    fragment_lengths = np.concatenate((region_fors[:,1] - region_fors[:,0],
                                       region_revs[1,:] - region_revs[0,:])).astype(int)

    return region_revs, region_fors, region

def calculate_bead_lims(bead_size, region_revs, region_fors):
    """
    Divides a region on a chromosome (or rather, the part of it covered by complete
    restriction fragments) into segments of equal, given length and one last
    segment which is smaller than the others such that the segments completely
    cover the region. These segments will be represented by spherical beads later.
    Returns the limits of the segments
    """
    region_length =   np.max((region_fors[-1,1], region_revs[1,-1])) \
                    - np.min((region_fors[0,0], region_revs[0,0]))
    n_beads = int(round(region_length / bead_size)) + 1
    bead_lims = [np.min((region_fors[0,0], region_revs[0,0])) + i * bead_size
                 for i in range(n_beads)]
    bead_lims[-1] = np.max((region_fors[-1,1], region_revs[1,-1]))

    return np.array(bead_lims)
    
def calculate_mappings(region_revs, region_fors, bead_lims):
    """
    Calculates a mapping assigning a bead to each restriction fragment.
    If one restriction fragment spans several beads, it will have the center
    bead (or center - 1 for even number of beads) assigned.
    Returns the mappings for reverse and forward restriction fragments
    """
    region_revs = region_revs.T

    mappings = []
    for rfs in (region_revs, region_fors):

        mapping = []
        for b, e in rfs:
            mapping.append((np.where(bead_lims <= b)[0][-1],
                                np.where(bead_lims <= e)[0][-1]))
        mapping = np.array(mapping)
        mapping = mapping.mean(1).astype(int)

        mappings.append(mapping)

    return mappings[0], mappings[1]

def build_cmatrix(rev_mapping, for_mapping, region, n_beads):
    """
    Builds a square contact frequency matrix of shape (n_beads, n_beads).
    Contacts from restriction fragments mapping to the same bead are summed.
    A zero in this matrix means either that there was no data collected or that
    the number of counts is in fact zero. Later, we ignore zero-valued
    entries in the matrix.
    Return square contact frequency matrix.
    """
    contmatrix = np.zeros((n_beads, n_beads))

    cmatrix = np.zeros((n_beads, n_beads))
    for i in range(n_beads):
        contributing_fors = np.where(for_mapping == i)[0]
        for j in range(n_beads):
            contributing_revs = np.where(rev_mapping == j)[0]
            for cfor in contributing_fors:
                for crev in contributing_revs:
                    cmatrix[i,j] += region[cfor, crev]
    cmatrix += cmatrix.T

    return cmatrix

def write_cmatrix(cmatrix, filename):
    """
    Writes a square contact frequency matrix to a file, which will be the
    input for our structure calculation code.
    """
    with open(filename, 'w') as opf:
        for i in range(len(cmatrix)):
            for j in range(i+1, len(cmatrix)):
                opf.write('{}\t{}\t{}\n'.format(i, j, int(cmatrix[i,j])))

## please change the path variable to a directory holding your Nora 2012 et al.
## raw data files

if True:
    ## male mESCs
    path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
    rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873934_male-mESCs-E14-replicate-1.matrix.txt')
    rev_fragments2, for_fragments2, matrix2 = parse_5C_file(path + 'GSM873935_male-mESCs-E14-replicate-2.matrix.txt')

if False:
    ## undifferentiated female mESCs
    path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
    rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873927_female-mESCs-PGK12.1-replicate-1.matrix.txt')
    rev_fragments2, for_fragments2, matrix2 = parse_5C_file(path + 'GSM873928_female-mESCs-PGK12.1-replicate-2.matrix.txt')

if False:
    ## female mESCs two days into differentiation
    path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
    rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873926_mESCs-female-PGK12.1-day2-Replicate1.txt')
    matrix2 = matrix1


## if there's two replicates, average counts
rev_fragments = rev_fragments1
for_fragments = for_fragments1
matrix = (matrix1 + matrix2) / 2.0

## set bead size in base pairs
bead_size = 15000

if not True:
    ## both TADs
    region_start = 100378306
    region_end = 101298738
    region_revs, region_fors, region = extract_region(matrix,
                                                      region_start, region_end)
    bead_lims = calculate_bead_lims(bead_size, region_revs, region_fors)
    n_beads = len(bead_lims)
    rev_mapping, for_mapping = calculate_mappings(region_revs, region_fors, bead_lims)
    cmatrix = build_cmatrix(rev_mapping, for_mapping, region, n_beads)
    write_cmatrix(cmatrix, path + '15kbbins_bothdomains.txt')

if not True:
    ## Tsix TAD
    region_start = 100378306
    region_end = 100699670
    region_revs, region_fors, region = extract_region(matrix,
                                                      region_start, region_end)
    bead_lims = calculate_bead_lims(bead_size, region_revs, region_fors)
    n_beads = len(bead_lims)
    rev_mapping, for_mapping = calculate_mappings(region_revs, region_fors, bead_lims)
    cmatrix = build_cmatrix(rev_mapping, for_mapping, region, n_beads)
    write_cmatrix(cmatrix, path + 'tsix.txt')
    
if not True:
    ## Xist TAD
    region_start = 100699670 + 1
    region_end = 101298738
    region_revs, region_fors, region = extract_region(matrix,
                                                      region_start, region_end)
    bead_lims = calculate_bead_lims(bead_size, region_revs, region_fors)
    n_beads = len(bead_lims)
    rev_mapping, for_mapping = calculate_mappings(region_revs, region_fors, bead_lims)
    cmatrix = build_cmatrix(rev_mapping, for_mapping, region, n_beads)
    write_cmatrix(cmatrix, path + 'xist.txt')

if True:
    ## prepare matrix in correct format for PGS (Alber lab)
    ## both TADs

    from ensemble_hic import kth_diag_indices    
    
    region_start = 100378306
    region_end = 101298738
    region_revs, region_fors, region = extract_region(matrix,
                                                      region_start, region_end)
    bead_lims = calculate_bead_lims(bead_size, region_revs, region_fors)
    n_beads = len(bead_lims)
    rev_mapping, for_mapping = calculate_mappings(region_revs, region_fors, bead_lims)
    cmatrix = build_cmatrix(rev_mapping, for_mapping, region, n_beads)

    ## make symmetric contact matrix with zero on diagonals and
    ## ones on side diagonals
    from scipy.spatial.distance import squareform
    cmatrix[np.diag_indices(len(cmatrix))] = 0
    flat = squareform(cmatrix)
    if True:
        ## for male mESCs, there are three data points which have
        ## exceedingly high counts. This clips these data points to the
        ## value of the 4th-highest counts
        th = flat[argsort(flat)][-4]
        flat[flat > th] = th
    flat /= flat.max()
    cmatrix = squareform(flat)
    cmatrix[kth_diag_indices(cmatrix, 1)] = 1.0
    cmatrix[kth_diag_indices(cmatrix, -1)] = 1.0

    ## Write data in format appropriate for the PGS code
    ## (https://github.com/alberlab/pgs/)
    ## please adapt paths and file names to your liking
    TAD_size = bead_size
    with open(os.path.expanduser('~/projects/mypgs/data/nora2012/prob_matrix_for_alber_final.txt'), 'w') as opf:
        for i, line in enumerate(cmatrix):
            opf.write('chr1\t{}\t{}'.format(region_start + i * TAD_size,
                                            region_start + (i + 1) * TAD_size))
            for x in line:
                opf.write('\t{:.6f}'.format(x))
            opf.write('\n')

    with open(os.path.expanduser('~/projects/ensemble_hic/data/nora2012/TADs_for_alber.txt'), 'w') as opf:
        for i in range(len(cmatrix)):
            opf.write('chr1\t{}\t{}\tdomain\n'.format(region_start + i * TAD_size,
                                                      region_start + (i+1) * TAD_size)
