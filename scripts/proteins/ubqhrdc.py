"""
Reconstruction of GB1 and Sh3 from mixed contacts
"""
import os, sys, numpy as np, glob

pypath = os.path.expanduser('~/projects/hic2/py')
os.chdir(pypath)
if not pypath in sys.path: sys.path.insert(0, pypath)

from csb.io import load
from csb.bio.io import StructureParser
from csb.bio.utils import distance_matrix
from prior import make_posterior, Contacts, MultiScaleModel
from csbplus.bio.dynamics import calc_distances
from scipy.spatial.distance import squareform
from isd import utils
from isd.Distance import Distance
from isd.DataSet import DataSet

pdbfile = '../data/1ubq.pdb'
chain   = StructureParser(pdbfile).parse().first_chain
coords  = chain.get_coordinates(['CA'])
seqs    = [chain.sequence]

pdbfile = '../data/2ma1.pdb'
chain   = StructureParser(pdbfile).parse().first_chain
coords2 = chain.get_coordinates(['CA'])
seqs   += [chain.sequence]

nbeads     = len(coords)
forcefield = 'rosetta'
factor     = (1.5,2.)[1]

k_nb = None 
k_bb = 250.

posterior = make_posterior(2*nbeads, forcefield, k_nb, k_bb)
universe  = posterior.universe
prior     = posterior.conformational_priors['tsallis_prior']
chain     = posterior.universe.get_molecule()
atomtype  = posterior.universe.atoms[0].type
beadsize  = float(prior.forcefield.d[atomtype,atomtype])

prior.beta = 1.
prior.E_min = -20.

state = posterior.as_state()

threshold  = factor * beadsize
distances  = distance_matrix(coords)
contacts   = np.transpose(np.nonzero(np.triu(distances<threshold,k=1)))
contacts   = np.sort(contacts,1).tolist()
distances2 = distance_matrix(coords2)
contacts2  = np.transpose(np.nonzero(np.triu(distances2<threshold,k=1)))
contacts2  = np.sort(contacts2,1).tolist()

contacts += contacts2

multiscale = MultiScaleModel(1)
model      = multiscale.create_model(nbeads)
restraints = model.make_contacts(posterior, contacts, spacing=threshold)

L = model.add_contact_likelihood(posterior, restraints, 'contacts')
posterior.add_likelihood(L)
L.error_model.sampler.update_alpha = 0
L.error_model.alpha = 10.

## no excluded volume interactions between copies

connectivity = universe.connectivity
connectivity[:nbeads,nbeads:] = 0
connectivity[nbeads:,:nbeads] = 0
universe.set_connectivity(connectivity)

## distances as ambiguous distance restraints

distances = []

for restraint in L.data:
    a = restraint.atom1[0]
    b = restraint.atom2[0]
    contributions = [(universe.atoms[a],universe.atoms[b]),
                     (universe.atoms[a+nbeads],universe.atoms[b+nbeads])]
    distances.append(Distance(restraint.value, contributions=contributions))

L.data = DataSet(distances)

## cut link between copies

bb = posterior.likelihoods['backbone']
restraints = []
for restraint in bb.data:
    a, b = restraint.contributions[0]
    i, j = sorted([a.index, b.index])
    if i == nbeads-1 and j == nbeads:
        print 'Skipping restraint:', i, j
        continue
    restraints.append(restraint)

bb.data = DataSet(restraints)

state = posterior.as_state()

posterior.energy(state)

run = 'ubqhrdc'
posterior.add_script('{}.py'.format(run))

if False:

    state.torsion_angles[:3*nbeads] = coords.flatten()
    state.torsion_angles[3*nbeads:] = coords2.flatten()

    from csb.bio.utils import rmsd, average_structure
    from csbplus.bio.dynamics import superimpose_ensemble, write_pdb
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    from mbo import srmsd
    from csb.bio.utils import rmsd, fit_wellordered, radius_of_gyration
    from scipy.cluster.vq import kmeans2
    
    pymol = Viewer()
    pymol.pymol_settings += ('split_chains models',
                             'set cartoon_trace_atoms=0',
                             'set grid_mode=1',
                             'load ../data/1ubq.pdb',
                             'load ../data/2ma1.pdb',
                             'order models_A 1ubq models_B 2ma1',
                             'remove !(name CA)',
                             'as ribbon',
                             'align 1ubq, models_A',
                             'align 2ma1, models_B',
                             'zoom all')
    burnin = 50
    E = utils.Load('~/tmp/{}_0'.format(run))
    E = E[burnin:]

    def assign(model, coords=coords, coords2=coords2):

        r = np.array([srmsd(coords,model), srmsd(-coords,model), 
                      srmsd(coords2,model), srmsd(-coords2,model)])
        i = r.argmin()

        return i/2, ['+','-'][i%2]

    classes = {(0,'+'): [], (0,'-'): [], (1,'+'): [], (1,'-'): []}

    for x in E.torsion_angles:
        x = x.reshape(2,nbeads,3)
        a = assign(x[0])
        b = assign(x[1])
        if a[0] != b[0]:
            classes[a].append(x[0])
            classes[b].append(x[1])

    means = [average_structure(classes[(0,'+')]),
             average_structure(classes[(1,'+')])]

    means[0] *= radius_of_gyration(coords) / radius_of_gyration(means[0])
    means[1] *= radius_of_gyration(coords2) / radius_of_gyration(means[1])

    pymol(means, sequences=seqs)
    
    x = average_structure(E.torsion_angles.reshape(-1,nbeads,3))

    print srmsd(coords,x),srmsd(-coords,x)

    x = coords

    K = 2
    m = kmeans2(r,K)[1]

    ensembles = [None,None]
    means = [None,None]

    for k in range(K):

        Y = np.compress(m==k,E.torsion_angles,0).reshape(-1,nbeads,3)
        y = superimpose_ensemble(Y,global_fit=False,reference=x)
        y = Y.mean(0)
        y*= radius_of_gyration(x) / radius_of_gyration(y)
        R, t = fit_wellordered(x,y)
        y = np.dot(y,R.T) + t
        index = 0
        
        if rmsd(x,Y[0]) > rmsd(-x,Y[0]):
            y = superimpose_ensemble(Y,global_fit=False)#,reference=-x)
            y = Y.mean(0)
            y*= radius_of_gyration(x) / radius_of_gyration(y)
            R, t = fit_wellordered(-x,y)
            y = np.dot(y,R.T) + t

            print 'reflection'
            index = 1
            
        print k, rmsd(x,y), rmsd(-x,y)

        ensembles[index] = Y
        means[index] = y

    ## compute violations for individual data sets and structures

    A = np.array(contacts[:-len(contacts2)])
    B = np.array(contacts2)

    if False:
        a = set(map(tuple,A.tolist())) - set(map(tuple,B.tolist()))
        b = set(map(tuple,B.tolist())) - set(map(tuple,A.tolist()))

        A = np.transpose(list(a))
        B = np.transpose(list(b))
    
    else:
        A = A.T
        B = B.T
        
    threshold = factor * beadsize * 1.02

    X = E.torsion_angles.reshape(len(E),2,nbeads,3)

    D = []
    for Y in X:
        v = []
        for y in Y:
            d = np.sum((y[A[0]] - y[A[1]])**2,1)**0.5
            v.append(np.mean(d<threshold))
        for y in Y:
            d = np.sum((y[B[0]] - y[B[1]])**2,1)**0.5
            v.append(np.mean(d<threshold))
        D.append(v)
    D = np.array(D)
