import numpy as np

def load_samples(samples_folder, n_replicas, n_samples,
                 dump_interval, burnin, interval=1):
    """Loads full results of a Replica Exchange
    simulation.

    :param samples_folder: directory in which samples are stored
    :type samples_folder: str ending with a slash ("/")

    :param n_replicas: number of replicas
    :type n_replicas: int

    :param n_samples: number of samples
    :type n_samples: int

    :param dump_interval: number of MCMC steps after which samples are written
    :type dump_interval: int

    :param burnin: number of MCMC samples discarded as burnin
    :type burnin: int (multiple of dump_interval)

    :param interval: return only every interval-th sample
    :type interval: int

    :returns: a two-dimensional array of MCMC samples with the first axis being
              the replicas and the second axis the samples for a given replica
    :rtype: :class:`numpy.ndarray`
    """

    samples = []
    for i in xrange(1, n_replicas + 1):
        samples.append(load_sr_samples(samples_folder, i, n_samples, dump_interval,
                                       burnin, interval))
        
    return np.array(samples)

def load_sr_samples(samples_folder, replica_id, n_samples,
                    dump_interval, burnin, interval=1):
    """Loads results for a single replica resulting from a Replica Exchange
    simulation.

    :param samples_folder: directory in which samples are stored
    :type samples_folder: str ending with a slash ("/")

    :param replica_id: number of replica of interest, 1-based indexing
    :type replica_id: int

    :param n_samples: number of samples
    :type n_samples: int

    :param dump_interval: number of MCMC steps after which samples are written
    :type dump_interval: int

    :param burnin: number of MCMC samples discarded as burnin
    :type burnin: int (multiple of dump_interval)

    :param interval: return only every interval-th sample
    :type interval: int

    :returns: an array of MCMC samples
    :rtype: :class:`numpy.ndarray`
    """

    samples = []
    for j in xrange((burnin / dump_interval) * dump_interval,
                    n_samples - dump_interval, dump_interval):
        path = samples_folder + 'samples_replica{}_{}-{}.pickle'
        samples += np.load(path.format(replica_id, j, j+dump_interval), allow_pickle=True)

    start = burnin * (burnin < dump_interval) + (burnin > dump_interval) * (burnin % dump_interval)
    
    return np.array(samples[start::interval])

def write_ensemble(X, filename, mol_ranges=None, center=True):
    """Writes a structure ensemble to a PDB file.

    :param X: coordinates of a structure ensemble
    :type X: :class:`numpy.ndarray`

    :param filename: file name of output PDB file
    :type filename: str

    :param mol_ranges: if writing structures consisting of several molecules, this
                       specifies start and end beads of the single molecules. Example:                       [0, 9, 19] for two molecules of 10 beads each
    :type mol_ranges: :class:`numpy.ndarray`

    :param center: if True, aligns the centers of mass of all structures
    :type center: bool
    """ 

    from csb.bio.structure import Atom, ProteinResidue, Chain, Structure, Ensemble
    from csb.bio.sequence import ProteinAlphabet

    if center:
        X -= X.mean(1)[:,None,:]

    if mol_ranges is None:
        mol_ranges = np.array([0, X.shape[1]])
    
    ensemble = Ensemble()

    for i, x in enumerate(X):
        structure = Structure('')
        structure.model_id = i + 1

        mol_coordinates = np.array([x[start:end]
                                    for start, end in zip(mol_ranges[:-1],
                                                          mol_ranges[1:])])
        for j, mol in enumerate(mol_coordinates):
            structure.chains.append(Chain(chr(65 + j)))

            for k, y in enumerate(mol):
                atom = Atom(k+1, 'CA', 'C', y)
                residue = ProteinResidue(k, 'ALA')
                residue.atoms.append(atom)
                structure.chains[chr(65 + j)].residues.append(residue)

        ensemble.models.append(structure)
    ensemble.to_pdb(filename)

def write_VMD_script(ensemble_pdb_file, bead_radii, output_file):
    """Writes a VMD script to show structures

    This writes a VMD script loading a structure ensemble PDB file, setting
    bead radii to given values and showing the structures as a chain of beads.

    :param ensemble_pdb_file: path to PDB file
    :type ensemble_pdb_file: str

    :param bead_radii: bead radii
    :type bead_radii: list-like of floats, length: # beads

    :param output_file: output file name
    :type output_file: str
    """

    lines = ['color Display Background white',
             'menu main on',
             'menu graphics on',
             'mol load pdb {}'.format(ensemble_pdb_file),
             'mol color Index',
             'mol delrep 0 0',
             'mol representation VDW',
             'mol addrep 0'
            ]

    radii_set = set(bead_radii)
    for br in radii_set:
        p1 = 'set sel [atomselect top "index '
        p2 = ''
        for i, r in enumerate(bead_radii):
            if r == br:
                p2 += '{} '.format(i)
        p3 = '"]'
        lines.append(p1 + p2[:-1] + p3)
        lines.append('$sel set radius {}'.format(br))
    with open(output_file,'w') as opf:
        [opf.write(line + '\n') for line in lines]


def write_pymol_script(ensemble_pdb_file, bead_radii, output_file,
                       repr='spheres', grid=False):
    """Writes a PyMol script to show structures

    This writes a PyMol script loading a structure ensemble PDB file, setting
    bead radii to given values and showing the structures as a chain of beads.
    .. warning:: I'm not sure whether this works (I mostly use VMD)

    :param ensemble_pdb_file: path to PDB file
    :type ensemble_pdb_file: str

    :param bead_radii: bead radii
    :type bead_radii: :class:`numpy.ndarray`

    :param output_file: output file name
    :type output_file: str

    :param repr: representation to use
    :type repr: str, either 'spheres', 'cartoon', or 'ribbon'

    :param grid: if True, show as grid
    :type grid: bool
    """

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
    """Loads results of a simulation using a config file

    This returns posterior samples from a simulation given a config file.

    :param config_file: path to config file
    :type config_file: str

    :param burnin: number of MCMC samples to be discarded as burnin
    :type burnin: int

    :returns: posterior samples
    :rtype: :class:`numpy.ndarray`
    """

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

def load_samples_from_cfg_auto(config_file, burnin=35000):
    """Loads results of a simulation using a config file

    This returns posterior samples from a simulation given a config file
    and automatically determines the number of actually drawn samples,
    i.e., it ignores to the n_samples setting in the config file.

    :param config_file: path to config file
    :type config_file: str

    :param burnin: number of MCMC samples to be discarded as burnin
    :type burnin: int

    :returns: posterior samples
    :rtype: :class:`numpy.ndarray`
    """

    import os
    from .setup_functions import parse_config_file

    cfg = parse_config_file(config_file)
    output_folder = cfg['general']['output_folder']
    n_structures = int(cfg['general']['n_structures'])
    n_replicas = int(cfg['replica']['n_replicas'])
    dump_interval = int(cfg['replica']['samples_dump_interval'])
    
    n_drawn_samples = 0
    fname =   output_folder + 'samples/samples_replica' \
            + str(n_replicas) + '_{}-{}.pickle'
    while True:
        if os.path.exists(fname.format(n_drawn_samples,
                                       n_drawn_samples + dump_interval)):
            n_drawn_samples += dump_interval
        else:
            break
    
    samples = load_sr_samples(output_folder + 'samples/',
                              n_replicas,
                              n_drawn_samples + 1,
                              dump_interval,
                              burnin)

    return samples

def load_ensemble_from_pdb(filename):
    """Loads a structure ensemble from a PDB file

    :param filename: file name of PDB file
    :type filename: str

    :returns: atom coordinates of structure ensemble
    :rtype: :class:`numpy.ndarray`
    """
    if False:
        ## Insanely slow
        from csb.bio.io.wwpdb import StructureParser
        ensemble = StructureParser(filename).parse_models()

        return np.array([m.get_coordinates(['CA']) for m in ensemble])
    else:
        ## Hacky
        ip = open(filename)
        lines = ip.readlines()
        ip.close()
        nres = 0
        for l in lines:
            w = l[:4]
            if nres > 0 and w != 'ATOM':
                break
            if l[:4] == 'ATOM':
                nres += 1

        import re
        atoms = []
        for l in lines:
            if 'ATOM' in l: atoms.append(l)
        atoms = [x.split() for x in atoms]
        atoms = [x[6:9] for x in atoms]
        atoms = np.array([np.array(x).astype(np.float) for x in atoms])
        atoms = np.array(np.split(atoms, nres))
        atoms = atoms.reshape(len(filter(lambda x: 'MODEL' in x, lines)), -1, 3)

        return atoms
   
def calculate_KL((posterior_distances, prior_distances, bins)):

    from csb.statistics.pdf import Gamma
    from csb.numeric import log

    g = Gamma()
    g.estimate(prior_distances)
    posterior_hist = np.histogram(posterior_distances, bins=bins, normed=True)[0]

    return np.trapz(posterior_hist * log(posterior_hist / g(bins[:-1])), bins[:-1])

def calculate_KL_KDE((posterior_distances, prior_distances)):

    from sklearn.neighbors import KernelDensity
    from scipy.integrate import quad

    h_silverman = lambda d: d.std() * (4. / 3 / len(d)) ** (1. / 5)
    h = h_silverman

    prior = KernelDensity(kernel='gaussian',
                          bandwidth=h(prior_distances)).fit(prior_distances.reshape(-1,1))
    posterior = KernelDensity(kernel='gaussian',
                              bandwidth=h(posterior_distances)).fit(posterior_distances.reshape(-1,1))

    ce = lambda x: -prior.score(x) * np.exp(posterior.score(x))
    hh = lambda x: -posterior.score(x) * np.exp(posterior.score(x))

    x_max = np.max((posterior_distances.max(), prior_distances.max()))
    vals = (quad(ce, 0., x_max)[0], quad(hh, 0., x_max)[0])

    return vals[0] - vals[1]


def calculate_KL_KDE_log((posterior_distances, prior_distances)):

    from sklearn.neighbors import KernelDensity
    from scipy.integrate import quad

    posterior_distances = np.log(posterior_distances)
    prior_distances = np.log(prior_distances)

    h_silverman = lambda d: d.std() * (4. / 3 / len(d)) ** (1. / 5)
    h = h_silverman

    prior = KernelDensity(kernel='gaussian',
                          bandwidth=h(prior_distances)).fit(prior_distances.reshape(-1,1))
    posterior = KernelDensity(kernel='gaussian',
                              bandwidth=h(posterior_distances)).fit(posterior_distances.reshape(-1,1))

    ce = lambda x: -prior.score(x) * np.exp(posterior.score(x))
    hh = lambda x: -posterior.score(x) * np.exp(posterior.score(x))

    vals = (quad(ce, -np.inf, np.inf)[0], quad(hh, -np.inf, np.inf)[0])

    return vals[0] - vals[1]

def calculate_DOS(config_file, n_samples, subsamples_fraction, burnin,
                  n_iter=100000, tol=1e-10, save_output=True, output_suffix=''):
    """Calculates the density of states (DOS) using non-parametric
    histogram reweighting (WHAM).

    :param config_file: Configuration file
    :type config_file: str

    :param n_samples: number of samples the simulation ran
    :type n_samples: int

    :param subsamples_fraction: faction of samples (after burnin) to be analyzed
                         set this to, e.g., 10 to use one tenth of
                         n_samples to decrease compution time
    :type subsamples_fraction: int

    :param burnin: number of samples to be thrown away as part
                   of the burn-in period
    :type burnin: int

    :param n_iter: number of WHAM iterations
    :type n_iter: int

    :param tol: threshold up to which the negative log-likelihood being minimized
                in WHAM can change before iteration stops
    :type tol: float

    :param save_output: save resulting DOS object, parameters used during
                        calculation and indices of randomly chosen samples
                        in simulation output folder
    :type save_output: True

    :returns: DOS object
    :rtype: DOS
    """
    
    from ensemble_hic.wham import PyWHAM as WHAM, DOS

    from ensemble_hic.setup_functions import parse_config_file, make_posterior
    from ensemble_hic.analysis_functions import load_sr_samples

    settings = parse_config_file(config_file)
    n_replicas = int(settings['replica']['n_replicas'])
    target_replica = n_replicas

    params = {'n_samples': n_samples,
              'burnin': burnin,
              'subsamples_fraction': subsamples_fraction,
              'niter': n_iter,
              'tol': tol
              }

    n_samples = min(params['n_samples'], int(settings['replica']['n_samples']))
    dump_interval = int(settings['replica']['samples_dump_interval'])

    output_folder = settings['general']['output_folder']
    if output_folder[-1] != '/':
        output_folder += '/'
    n_beads = int(settings['general']['n_beads'])
    n_structures = int(settings['general']['n_structures'])
    schedule = np.load(output_folder + 'schedule.pickle')

    posterior = make_posterior(settings)
    p = posterior
    variables = p.variables

    energies = []
    L = p.likelihoods['ensemble_contacts']
    data = L.forward_model.data_points
    P = p.priors['nonbonded_prior']
    sels = []
    for i in range(n_replicas):
        samples = load_sr_samples(output_folder + 'samples/', i+1, n_samples+1,
                                  dump_interval, burnin=params['burnin'])
        sel = np.random.choice(len(samples),
                               int(len(samples) / float(subsamples_fraction)),
                               replace=False)
        samples = samples[sel]
        sels.append(sel)
        energies.append([[-L.log_prob(**x.variables) if 'lammda' in schedule else 0,
                          -P.log_prob(structures=x.variables['structures'])
                          if 'beta' in schedule else 0]
                         for x in samples])
        print "Calculated energies for {}/{} replicas...".format(i, n_replicas)
        
    energies = np.array(energies)
    energies_flat = energies.reshape(np.prod(energies.shape[:2]), 2)
    sched = np.array([schedule['lammda'], schedule['beta']])
    q = np.array([[(energy * replica_params).sum() for energy in energies_flat]
                     for replica_params in sched.T])
    wham = WHAM(len(energies_flat), n_replicas)
    wham.N[:] = len(energies_flat)/n_replicas
    wham.run(q, niter=params['niter'], tol=params['tol'], verbose=100)

    dos = DOS(energies_flat, wham.s, sort_energies=False)

    if save_output:
        import os
        import sys
        from cPickle import dump

        ana_path = output_folder + 'analysis/'
        if not os.path.exists(ana_path):
            os.makedirs(ana_path)
        with open(ana_path + 'dos{}.pickle'.format(output_suffix), 'w') as opf:
            dump(dos, opf)
        with open(ana_path + 'wham_params{}.pickle'.format(output_suffix), 'w') as opf:
            dump(params, opf)
        with open(ana_path + 'wham_sels{}.pickle'.format(output_suffix), 'w') as opf:
            dump(np.array(sels), opf)

    return dos

def calculate_evidence(dos):
    """Calculates the evidence from a DOS object

    :param dos: DOS object (output from calculate_DOS)
    :type dos: DOS
    :returns: log-evidence (without additive constants stemming from likelihood
              normalization)
    :rtype: float
    """
    
    from csb.numeric import log_sum_exp

    return log_sum_exp(-dos.E.sum(1) + dos.s) - \
           log_sum_exp(-dos.E[:,1] + dos.s)
