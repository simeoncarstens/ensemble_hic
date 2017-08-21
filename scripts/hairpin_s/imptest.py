from __future__ import print_function

import IMP 
import IMP.algebra 
import IMP.core
import IMP.container
import IMP.atom
import IMP.isd
m = IMP.Model() 

use_nbl = not True
timestep = 0.7
nsteps = 20
nparticles = 1

particles = IMP.core.create_xyzr_particles(m, nparticles, 1.0)
ps_container = IMP.container.ListSingletonContainer(m, particles)

delta = 2.0
for i, p in enumerate(particles):
    p.set_coordinates(IMP.algebra.Vector3D(float(i) * delta, 0.0, 0.0)) 

restraints = []

# Set up excluded volume
if use_nbl:
    close_pairs = IMP.container.ClosePairContainer(particles, 0.0, 15.0)
else:
    close_pairs = IMP.container.AllPairContainer(particles)
sdps = IMP.core.SoftSpherePairScore(1)
evr = IMP.container.PairsRestraint(sdps, close_pairs)
 
bonded_particles = [IMP.atom.Bonded.setup_particle(m, ps_container.get_indexes()[i])
                    for i in range(len(particles))]
for i in range(len(particles) - 1):
    IMP.atom.create_custom_bond(bonded_particles[i], bonded_particles[i+1], 2.0)

if use_nbl:
    nbl.add_pair_filter(IMP.atom.BondedPairFilter())

restraints = [evr]

sf = IMP.core.RestraintsScoringFunction(restraints)
for p in particles:
    p.set_coordinates_are_optimized(True)
if True:
    for p in particles:
        IMP.atom.Mass.setup_particle(p, 1.0)
    
    o = IMP.isd.HybridMonteCarlo(m, 1.0, nsteps, timestep)
    o.set_was_used(True)
    o.get_md().set_scoring_function(sf)
else:
    o = IMP.core.MonteCarlo(m)
    for i in range(len(particles)):
        o.add_mover(IMP.core.NormalMover(ps_container.get_particles(),
                                         IMP.core.XYZ.get_xyz_keys(), 1.0))
o.set_scoring_function(sf)


def sample(n):
    coords = []
    vels = []
    for _ in range(n):
        o.get_md().optimize(0)
        o.optimize(1)
        coords.append([p.get_coordinates() for p in particles])
        vels.append([p.get_value(IMP.FloatKey("vx")) for p in particles])

    return coords, vels

X, V = numpy.array(sample(10000))
X = numpy.array([numpy.array(x[0]) for x in X])
V = V.astype(float)
o.optimize(1000)
