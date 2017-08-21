import numpy as np

def add_leftout_RFs(data_file, ignored_rfs):

    for (start, end, n) in ignored_rfs:
        if n > 1:
            interp = np.linspace(start, end, n+1)
            starts = interp[:-1, None]
            ends = interp[1:, None]
            ns = np.ones(n)[:,None]
            im_frags = np.hstack((starts, ends, ns)).astype(int)
            ignored_rfs = np.vstack((ignored_rfs, 
                                        im_frags))
    ignored_rfs = ignored_rfs[ignored_rfs[:,2] == 1]

    data = np.loadtxt(data_file)
    data = np.vstack((data, np.hstack((ignored_rfs[:,:2], 
                                       -np.ones((len(ignored_rfs), 
                                                 data.shape[1] - 2)).astype(int)))))
    
    return data.astype(int)

def make_beads_RF_list(data_matrix):

    rv_RFs = data_matrix[:2,2:].T
    fw_RFs = data_matrix[2:,:2]
    beads = np.vstack((fw_RFs, rv_RFs)).astype(int)
    a = np.hstack((beads, np.arange(len(beads))[None,:].T))
    a = a[a[:,1].argsort()]
    fw_indices = np.array([np.where(a[:,:2] == x)[0][0] for x in fw_RFs])
    rv_indices = np.array([np.where(a[:,:2] == x)[0][0] for x in rv_RFs])

    return np.hstack((fw_indices[None,:].T, fw_RFs)), np.hstack((rv_indices[None,:].T, rv_RFs))

def make_data(data_file):

    ignored_rfs = np.array([[5693, 11138, 4],
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

    extended_data_matrix = add_leftout_RFs(data_file, ignored_rfs)
    fw_beads_RFs, rv_beads_RFs = make_beads_RF_list(extended_data_matrix)
    fw_indices = fw_beads_RFs[:,0]
    rv_indices = rv_beads_RFs[:,0]
    
    counts = extended_data_matrix[2:,2:]
    data = [[counts[i,j], fw_indices[i], rv_indices[j]]
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


if __name__ == '__main__':

    import os

    data_set = 'GM12878'
    # data_set = 'K562'
    
    data_path = os.path.expanduser('~/projects/ensemble_hic/data/bau2011/')
    data, extended_5C_matrix = make_data(data_path + '{}.txt'.format(data_set))
    bead_radii = calculate_bead_radii(extended_5C_matrix)

    np.savetxt(data_path + '{}_processed.txt'.format(data_set), data, fmt='%i')

    this_dir = os.path.dirname(os.path.abspath(__file__))
    np.savetxt(this_dir + '/bead_radii.txt', bead_radii)
