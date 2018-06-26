import os
import numpy as np

def parse_5C_file(filename):

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

    region_length =   np.max((region_fors[-1,1], region_revs[1,-1])) \
                    - np.min((region_fors[0,0], region_revs[0,0]))
    n_beads = int(round(region_length / bead_size)) + 1
    bead_lims = [np.min((region_fors[0,0], region_revs[0,0])) + i * bead_size
                 for i in range(n_beads)]
    bead_lims[-1] = np.max((region_fors[-1,1], region_revs[1,-1]))

    return np.array(bead_lims)
    
def calculate_mappings(region_revs, region_fors, bead_lims):

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
    
    with open(filename, 'w') as opf:
        for i in range(len(cmatrix)):
            for j in range(i+1, len(cmatrix)):
                opf.write('{}\t{}\t{}\n'.format(i, j, int(cmatrix[i,j])))

# path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
# rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873934_male-mESCs-E14-replicate-1.matrix.txt')
# rev_fragments2, for_fragments2, matrix2 = parse_5C_file(path + 'GSM873935_male-mESCs-E14-replicate-2.matrix.txt')

# path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
# rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873927_female-mESCs-PGK12.1-replicate-1.matrix.txt')
# rev_fragments2, for_fragments2, matrix2 = parse_5C_file(path + 'GSM873928_female-mESCs-PGK12.1-replicate-2.matrix.txt')

path = os.path.expanduser('~/projects/ensemble_hic/data/nora2012/')
rev_fragments1, for_fragments1, matrix1 = parse_5C_file(path + 'GSM873926_mESCs-female-PGK12.1-day2-Replicate1.txt')
matrix2 = matrix1


rev_fragments = rev_fragments1
for_fragments = for_fragments1
matrix = (matrix1 + matrix2) / 2.0

bead_size = 3000

if True:
    ## both TADs
    region_start = 100378306
    region_end = 101298738
    region_revs, region_fors, region = extract_region(matrix,
                                                      region_start, region_end)
    bead_lims = calculate_bead_lims(bead_size, region_revs, region_fors)
    n_beads = len(bead_lims)
    rev_mapping, for_mapping = calculate_mappings(region_revs, region_fors, bead_lims)
    cmatrix = build_cmatrix(rev_mapping, for_mapping, region, n_beads)
    write_cmatrix(cmatrix, path + 'female_day2_bothdomains.txt')

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
