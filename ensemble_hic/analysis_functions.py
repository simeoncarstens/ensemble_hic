import numpy as np

def load_samples(samples_folder, n_replicas, n_samples,
                 dump_interval, burnin, interval=1):

    samples = []
    for i in xrange(1, n_replicas + 1):
        samples.append(load_sr_samples(samples_folder, i, n_samples, dump_interval,
                                       burnin, interval))
        
    return np.array(samples)

def load_sr_samples(samples_folder, replica_id, n_samples,
                    dump_interval, burnin, interval=1):

    samples = []
    for j in xrange((burnin / dump_interval) * dump_interval,
                    n_samples - dump_interval, dump_interval):
        path = samples_folder + 'samples_replica{}_{}-{}.pickle'
        samples += np.load(path.format(replica_id, j, j+dump_interval))

    start = burnin * (burnin < dump_interval) + (burnin > dump_interval) * (burnin % dump_interval)
    
    return np.array(samples[start::interval])

def write_ensemble(X, filename, center=True):

    from csb.bio.structure import Atom, ProteinResidue, Chain, Structure, Ensemble
    from csb.bio.sequence import ProteinAlphabet

    if center:
        X -= X.mean(1)[:,None,:]
    
    ensemble = Ensemble()

    for i, x in enumerate(X):
        structure = Structure('')
        structure.model_id = i + 1
        structure.chains.append(Chain('A'))
        x = x.reshape(-1, 3)

        for k, y in enumerate(x):
            atom = Atom(k+1, 'CA', 'C', y)
            residue = ProteinResidue(k, 'ALA')
            residue.atoms.append(atom)
            structure.chains['A'].residues.append(residue)

        ensemble.models.append(structure)
    ensemble.to_pdb(filename)

def write_VMD_script(ensemble_pdb_file, bead_radii, output_file):

    lines = ['color Display Background white',
             'menu main on',
             'menu graphics on',
             'mol load pdb {}'.format(ensemble_pdb_file),
             'mol color Index',
             'mol delrep 0 0',
             'mol representation VDW',
             'mol addrep 0'
            ]

    for i, r in enumerate(bead_radii):
        lines.append('set sel [atomselect top "index {}"]'.format(i))
        lines.append('$sel set radius {}'.format(r))

    with open(output_file,'w') as opf:
        [opf.write(line + '\n') for line in lines]



def write_pymol_script(ensemble_pdb_file, bead_radii, output_file,
                       repr='spheres', grid=False):

    epf = ensemble_pdb_file
    fname = epf[-epf[::-1].find('/'):epf.find('.pdb')]
    lines = ['load {}'.format(ensemble_pdb_file),
             'hide all',
             'util.chainbow',
             'set cartoon_trace_atoms=1',
             'set ribbon_trace_atoms=1',
            ]
    if repr == 'spheres':
        for i, r in enumerate(bead_radii):
            lines.append('set sphere_scale={}, resi {}'.format(r / 4.0, i))
    elif repr == 'cartoon':
        lines.append('as cartoon')
    elif repr == 'ribbon':
        lines.append('as ribbon')
    if grid:
        lines.append('set grid_mode=1')
        lines.append('hide {}'.format(fname))

    with open(output_file,'w') as opf:
        [opf.write(line + '\n') for line in lines]


def load_samples_from_cfg(config_file, burnin=35000):

    from .setup_functions import parse_config_file

    cfg = parse_config_file(config_file)
    output_folder = cfg['general']['output_folder']
    n_beads = int(cfg['general']['n_beads'])
    n_structures = int(cfg['general']['n_structures'])
    n_samples = int(cfg['replica']['n_samples'])
    samples = load_sr_samples(output_folder + 'samples/',
                              int(cfg['replica']['n_replicas']),
                              n_samples,
                              int(cfg['replica']['samples_dump_interval']),
                              burnin)

    return samples
