import numpy as np
import os

ignored_rfs = np.array([[5693, 11138, 1],
                        [109838, 120933, 1],
                        [134334, 147782, 1],
                        [189074, 192185, 1],
                        [235762, 237363, 1],
                        [247214, 260279, 1],
                        [277942, 286059, 1],
                        [327240, 331940, 1],
                        [335401, 346310, 1],
                        [372670, 376971, 1],
                        [401140, 402437, 1],
                        [433293, 441045, 1]])

no_data_rfs = np.array([[74448, 86128, 1],
                        [225341, 235762, 1],
                        [380160, 401140, 1]])
                        

def add_leftout_RFs(data_file, ignored_rfs, no_data_rfs):

    rfs_to_be_added = np.concatenate((ignored_rfs, no_data_rfs))
    for (start, end, n) in rfs_to_be_added:
        if n > 1:
            interp = np.linspace(start, end, n+1)
            starts = interp[:-1, None]
            ends = interp[1:, None]
            ns = np.ones(n)[:,None]
            im_frags = np.hstack((starts, ends, ns)).astype(int)
            ignored_rfs = np.vstack((ignored_rfs, 
                                        im_frags))
    rfs_to_be_added = rfs_to_be_added[rfs_to_be_added[:,2] == 1]

    data = np.loadtxt(data_file)
    data = np.vstack((data, np.hstack((rfs_to_be_added[:,:2], 
                                       -np.ones((len(rfs_to_be_added), 
                                                 data.shape[1] - 2)).astype(int)))))
    
    return data.astype(int)

def make_beads_RF_list(data_matrix):

    rv_RFs = data_matrix[:2,2:].T
    fw_RFs = data_matrix[2:,:2]
    ## add missing 23 base pairs between fw fragment 8 and
    ## rv fragment 10 in Supplementary Table 1 from Bau et al. (2011)
    ## to fw fragment 8
    fw_RFs[fw_RFs[:,1] == 29756,1] = 29779
    
    beads = np.vstack((fw_RFs, rv_RFs)).astype(int)
    a = np.hstack((beads, np.arange(len(beads))[None,:].T))
    a = a[a[:,1].argsort()]
    fw_indices = np.array([np.where(a[:,:2] == x)[0][0] for x in fw_RFs])
    rv_indices = np.array([np.where(a[:,:2] == x)[0][0] for x in rv_RFs])

    return np.hstack((fw_indices[None,:].T, fw_RFs)),\
           np.hstack((rv_indices[None,:].T, rv_RFs))

def make_data(data_file, ignored_rfs, no_data_rfs):

    extended_data_matrix = add_leftout_RFs(data_file, ignored_rfs, no_data_rfs)
    fw_beads_RFs, rv_beads_RFs = make_beads_RF_list(extended_data_matrix)
    fw_indices = fw_beads_RFs[:,0]
    rv_indices = rv_beads_RFs[:,0]
    
    counts = extended_data_matrix[2:,2:]
    data = [[fw_indices[i], rv_indices[j], counts[i,j]]
            for i in range(len(fw_indices))
            for j in range(len(rv_indices)) if counts[i,j] > -1]

    return np.array(data).astype(int), extended_data_matrix

def calculate_bead_radii(data_matrix):

    fw_beads_RFs, rv_beads_RFs = make_beads_RF_list(data_matrix)
    beads_RFs = np.vstack((fw_beads_RFs, rv_beads_RFs))
    sorted_beads_RFs = beads_RFs[beads_RFs[:,0].argsort()]

    radii = (sorted_beads_RFs[:,2] - sorted_beads_RFs[:,1]) ** 0.3333
    radii /= np.median(radii)
    return radii

map_pos_to_bead = lambda pos: map_pos_to_bead_general(pos, os.path.expanduser('~/projects/ensemble_hic/data/bau2011/K562.txt'), ignored_rfs, no_data_rfs)

def map_pos_to_bead_general(pos, data_file, ignored_rfs, no_data_rfs):
    ext_data_matrix = add_leftout_RFs(data_file, ignored_rfs, no_data_rfs)
    fw_beads_RFs, rv_beads_RFs = make_beads_RF_list(ext_data_matrix)
    all_beads_RFs = np.concatenate((fw_beads_RFs, rv_beads_RFs))

    return all_beads_RFs[np.logical_and(all_beads_RFs[:,1] <= pos,
                                        pos < all_beads_RFs[:,2]),0][0]


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


if __name__ == '__main__':

    import os

    data_set = 'GM12878'
    # data_set = 'K562'
    
    data_path = os.path.expanduser('~/projects/ensemble_hic/data/bau2011/')
    data, extended_5C_matrix = make_data(data_path + '{}.txt'.format(data_set),
                                         ignored_rfs, no_data_rfs)
    bead_radii = calculate_bead_radii(extended_5C_matrix)

    if not False:
        np.savetxt(data_path + '{}_processed_fixed.txt'.format(data_set),
                   data, fmt='%i')
        
        this_dir = os.path.dirname(os.path.abspath(__file__))
        np.savetxt(this_dir + '/bead_radii_fixed.txt', bead_radii)
