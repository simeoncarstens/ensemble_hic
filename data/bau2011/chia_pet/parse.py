import os, sys
import numpy as np

def parse_li2012(filename):
    ## filename: table S3 from "Extensive Promoter-Centered Chromatin Interactions
    ## Provide a Topological Basis for Transcription Regulation", Li et al., Cell 2012
    ## Genomic coordinates are in hg19

    from xlrd import open_workbook

    wb = open_workbook(filename)
    s1 = wb.sheets()[7]
    s2 = wb.sheets()[8]

    d1 = np.array([np.array(s1.row_values(i))[:7] for i in range(4, s1.nrows)])
    d2 = np.array([np.array(s2.row_values(i))[:7] for i in range(4, s2.nrows)])
    d = np.concatenate((d1, d2))
    chr16 = d[np.logical_and(d[:,0] == 'chr16', d[:,3] == 'chr16')]
    chr16 = chr16[:,(1,2,4,5,6)].astype(float).astype(int)

    chr16_hg18 = convert_hg19_to_hg18(chr16)
    lim = 500000
    ENm008_hg18 = chr16_hg18[np.all(chr16_hg18[:,:4] <= lim, 1)]

    return ENm008_hg18


def parse_heidari2014(*filenames):

    d = np.concatenate([np.loadtxt(fn, dtype=str, skiprows=1)
                        for fn in filenames])
    d = d[d[:,0] == 'chr16']
    d = d[d[:,3] == 'chr16']
    d = d[:,(1,2,4,5,12)].astype(int)

    hg18 = convert_hg19_to_hg18(d)
    lim = 500000  ## right limit of ENm008 region
    
    return hg18[np.all(hg18[:,:4] <= lim, 1)]

def convert_hg19_to_hg18(interactions):

    from pyliftover import LiftOver

    lo = LiftOver('hg19', 'hg18')
    interactions_hg18 = []
    for x in interactions:
        y = []
        for i in range(4):
            new_coords = lo.convert_coordinate('chr16', x[i])
            if len(new_coords) > 1:
                raise
            if new_coords[0][0] == 'chr16':
                y.append(new_coords[0][1])
        interactions_hg18.append(y + [x[-1]])
    interactions_hg18 = np.array(interactions_hg18)

    return interactions_hg18


def map_contacts_to_beads(contacts, data_file=None):
    ppath = os.path.expanduser('~/projects/ensemble_hic/')
    if data_file is None:
        data_file = ppath + '/data/bau2011/K562.txt'
    sys.path.append(ppath + '/scripts/bau2011/')
    from process_5C_data import map_pos_to_bead, ignored_rfs, no_data_rfs

    contacts_beads = np.array([map(map_pos_to_bead, x[:4]) + [x[4]] for x in contacts])

    return contacts_beads    


def filter_not_in_5C(contact_beads, data_file=None):
    if data_file is None:
        data_file = os.path.expanduser('~/projects/ensemble_hic/data/bau2011/K562_processed_fixed.txt')
    fiveC_d = np.loadtxt(data_file).astype(int)
    fiveC_contacts = np.sort(fiveC_d[:,:2], axis=1)
    fiveC_d_sorted = np.hstack((fiveC_contacts, fiveC_d[:,2][:,None]))

    not_in_5C = []
    for i, x in enumerate(contact_beads):
        range1 = range(x[0], x[1]+1)
        range2 = range(x[2], x[3]+1)
        in_5C = False
        for j, y in enumerate(fiveC_d_sorted):
            if y[0] in range1 and y[1] in range2:
                in_5C = True
                break
        if not in_5C:
            not_in_5C.append(x)
    not_in_5C = np.array(not_in_5C)
    not_in_5C = np.array(filter(lambda x: min(x[2:4]) - max(x[:2]) > 2, not_in_5C))
    doubles = [i for i in range(len(not_in_5C) - 1)
               if np.all(not_in_5C[i,:4] == not_in_5C[i+1,:4])]
    not_in_5C = not_in_5C[[i for i in range(len(not_in_5C)) if not i in doubles]]

    return not_in_5C
