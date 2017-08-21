import numpy


def prepare_data(contacts_file, ignore_sequential_neighbors=2, disregard_lowest=0.0):

    data = numpy.loadtxt(contacts_file, int)
    data = data[:,(2,0,1)]
    data = data[numpy.abs(data[:,1] - data[:,2]) > ignore_sequential_neighbors]
    data = data[data[:,0] > 0]
    data = data[data[:,0].argsort(axis=0)][int(disregard_lowest * len(data)):]

    return data

def make_likelihood(n_structures, contacts_file, disregard_lowest=0.0,
                    contact_distance=1.5, lammda=1.0, ignore_sequential_neighbors=2):

    from hicisd2.forward_models.ensemble_contacts import EnsembleContactsFWM
    from hicisd2.error_models.ensemble_contacts import EnsembleContactsGaussianEM
    from hicisd2.likelihoods.ensemble_contacts import EnsembleContactsLikelihood

    data = prepare_data(contacts_file, ignore_sequential_neighbors, disregard_lowest)
        
    FWM = EnsembleContactsFWM('ensemble_contacts_fwm', n_structures,
                              contact_distance, cutoff=1000,
                              data_points=data)

    EM = EnsembleContactsGaussianEM('ensemble_contacts_em', data[:,0])
    CL = EnsembleContactsLikelihood('ensemble_contacts', fwm=FWM, em=EM,
                                    lammda=lammda)
    
    return CL

def make_poisson_likelihood(n_structures, contacts_file, disregard_lowest=0.0,
                            contact_distance=1.5, lammda=1.0,
                            ignore_sequential_neighbors=2):


    from hicisd2.forward_models.ensemble_contacts import EnsembleContactsFWM
    from hicisd2.error_models.ensemble_contacts import EnsembleContactsPoissonEM
    from hicisd2.likelihoods.ensemble_contacts import EnsembleContactsLikelihood

    data = prepare_data(contacts_file, ignore_sequential_neighbors, disregard_lowest)        
    FWM = EnsembleContactsFWM('ensemble_contacts_fwm', n_structures,
                              contact_distance, cutoff=1000,
                              data_points=data)
    EM = EnsembleContactsPoissonEM('ensemble_contacts_em', data[:,0])
    CL = EnsembleContactsLikelihood('ensemble_contacts', fwm=FWM, em=EM,
                                    lammda=lammda)
    
    return CL

def make_lognormal_likelihood(n_structures, contacts_file, disregard_lowest=0.0,
                              contact_distance=1.5, lammda=1.0,
                              ignore_sequential_neighbors=2):


    from hicisd2.forward_models.ensemble_contacts import EnsembleContactsFWM
    from hicisd2.error_models.ensemble_contacts import EnsembleContactsLognormalEM
    from hicisd2.likelihoods.ensemble_contacts import EnsembleContactsLikelihood

    data = prepare_data(contacts_file, ignore_sequential_neighbors, disregard_lowest)        
    FWM = EnsembleContactsFWM('ensemble_contacts_fwm', n_structures,
                              contact_distance, cutoff=1000,
                              data_points=data)
    EM = EnsembleContactsLognormalEM('ensemble_contacts_em', data[:,0])
    EM.targets = EM.data
    CL = EnsembleContactsLikelihood('ensemble_contacts', fwm=FWM, em=EM,
                                    lammda=lammda)
    
    return CL
    


def create_ensemble_contacts_posterior(n_beads, n_structures, likelihood, beta=1.0,
                                       k_bb=25.0, bead_diameter=1.0, k_ve=1.0):

    from hicisd2.priors.boltzmann import EnsembleContactsSimpleFFBoltzmannPrior
    from hicisd2.priors.backbone import EnsembleContactsBBHarmonicRestraintPrior
    from hicisd2.priors.jeffreys import DistanceScalePrior, SmoothSteepnessPrior, LowerUpperEMPrior
    from hicisd2.priors.rog import EnsembleGyrationRadiusPrior
    from hicisd2.priors.sphere import EnsembleSpherePrior
    from hicisd2.forcefields.nb import NBQuarticVEFF
    from hicisd2.forcefields.simple import cQuarticVEFF
    from hicisd2.hicisd2lib import create_backbone_restraints
    from isd2.pdf.posteriors import Posterior
    from hicisd2.priors.jeffreys import LognormalEMPrior
    
    bb_data, bb_ll, bb_ul = create_backbone_restraints(n_beads, bead_diameter)

    CL = likelihood
    
    FF = cQuarticVEFF('quartic_ve_ff', k1=k_ve, d0=bead_diameter)
    # FF = NBQuarticVEFF('quartic_ve_ff', k1=k_ve, d0=bead_diameter, n_beads=n_beads)
    P = EnsembleContactsSimpleFFBoltzmannPrior('boltzmann_prior', beta=beta,
                                               forcefield=FF, 
                                               n_structures=n_structures)
    BBP = EnsembleContactsBBHarmonicRestraintPrior('backbone_prior', bb_ll, bb_ul,
                                                   k_bb, n_structures)
    SP = EnsembleSpherePrior('sphere_prior', 6.0, 5.0, n_structures)
    priors = {'boltzmann_prior': P,
              'backbone_prior': BBP,
              'sphere_prior': SP
              }
    if 'k2' in CL.variables:
        k2P = LognormalEMPrior('k2_prior')
        priors.update(k2_prior=k2P)
    
    posterior = Posterior({'contacts': CL}, priors)

    return posterior

def make_posterior(n_structures, contacts_file, disregard_lowest=0.0, n_beads=23,
                   contact_distance=1.5, smooth_steepness=10, beta=1.0, lammda=1.0,
                   ignore_sequential_neighbors=2, k_ve=1.0, k2=1.0):

    likelihood = make_likelihood(n_structures, contacts_file,
                                 disregard_lowest=disregard_lowest,
                                 contact_distance=contact_distance,
                                 ignore_sequential_neighbors=ignore_sequential_neighbors)
    posterior = create_ensemble_contacts_posterior(n_beads=n_beads,
                                                   n_structures=n_structures,
                                                   likelihood=likelihood, k_ve=k_ve)
    posterior = posterior.conditional_factory(smooth_steepness=smooth_steepness, 
                                              distance_scale=1.0)
    posterior['lammda'].set(lammda)
    posterior['beta'].set(beta)

    return posterior


def make_poisson_posterior(n_structures, contacts_file, disregard_lowest=0.0, n_beads=23,
                           contact_distance=1.5, smooth_steepness=10,
                           beta=1.0, lammda=1.0, ignore_sequential_neighbors=2,
                           k_ve=1.0, k2=1.0):

    likelihood = make_poisson_likelihood(n_structures, contacts_file,
                                         disregard_lowest=disregard_lowest,
                                         contact_distance=contact_distance,
                                         ignore_sequential_neighbors=ignore_sequential_neighbors)
    posterior = create_ensemble_contacts_posterior(n_beads=n_beads,
                                                   n_structures=n_structures,
                                                   likelihood=likelihood, k_ve=k_ve)
    posterior = posterior.conditional_factory(smooth_steepness=smooth_steepness, 
                                              distance_scale=1.0)
    posterior['lammda'].set(lammda)
    posterior['beta'].set(beta)

    return posterior


def make_lognormal_posterior(n_structures, contacts_file, disregard_lowest=0.0, n_beads=23,
                             contact_distance=1.5, smooth_steepness=10,
                             beta=1.0, lammda=1.0, ignore_sequential_neighbors=2,
                             k_ve=1.0, k2=1.0):

    likelihood = make_lognormal_likelihood(n_structures, contacts_file,
                                           disregard_lowest=disregard_lowest,
                                           contact_distance=contact_distance,
                                           ignore_sequential_neighbors=ignore_sequential_neighbors)
    posterior = create_ensemble_contacts_posterior(n_beads=n_beads,
                                                   n_structures=n_structures,
                                                   likelihood=likelihood, k_ve=k_ve)
    posterior = posterior.conditional_factory(smooth_steepness=smooth_steepness, 
                                              distance_scale=1.0)
    posterior['lammda'].set(lammda)
    posterior['beta'].set(beta)

    return posterior



from isd2.samplers.hmc import ISD2FastHMCSampler, HMCSampleStats
class MyISD2FastHMCSampler(ISD2FastHMCSampler):

    def get_last_draw_stats(self):
        return HMCSampleStats(self.last_move_accepted, self._nmoves, self.timestep)

    
def make_subsamplers_weights(posterior, init_state,
                             structures_timestep=1e-4, structures_nsteps=100,
                             weights_timestep=1e1, weights_nsteps=50):

    from collections import OrderedDict
    from hicisd2.npsamplers import EnsembleContactsWeightsSampler
    from hicisd2.error_models.ensemble_contacts import EnsembleContactsGaussianEM, EnsembleContactsLognormalEM
    
    if 'k2' in init_state.variables:
        xpdf = posterior.conditional_factory(weights=init_state.variables['weights'],
                                             k2=init_state.variables['k2'])
    else:
        xpdf = posterior.conditional_factory(weights=init_state.variables['weights'])
    structure_sampler = MyISD2FastHMCSampler(xpdf,
                                             init_state.variables['structures'],
                                             structures_timestep, structures_nsteps)
    subsamplers = OrderedDict(structures=structure_sampler)
    
    weights_sampler = EnsembleContactsWeightsSampler(init_state.variables['weights'],
                                                     n_iterations=1,
                                                     n_steps=weights_nsteps,
                                                     timestep=weights_timestep)
    em = posterior.likelihoods['contacts'].error_model
    if 'k2' in init_state.variables:
        if isinstance(em, EnsembleContactsGaussianEM):
            from hicisd2.npsamplers import PrecisionSampler
            k2_sampler = PrecisionSampler(init_state=init_state.variables['k2'],
                                          variable_name='k2',
                                          adapt_stepsize=True, greater_zero=True)
        elif isinstance(em, EnsembleContactsLognormalEM):
            k2_sampler = LognormalGammaSampler(n_datapoints=len(posterior.likelihoods['contacts'].forward_model.data_points))
        k2_sampler.pdf = posterior.conditional_factory(structures=init_state.variables['structures'])
        subsamplers.update(k2=k2_sampler)
    subsamplers.update(weights=weights_sampler)
        
    return subsamplers


def make_subsamplers_onlystructures(posterior, init_state,
                                    structures_timestep=1e-4, structures_nsteps=100):

    from collections import OrderedDict
    from hicisd2.npsamplers import LognormalGammaSampler
    from hicisd2.error_models.ensemble_contacts import EnsembleContactsGaussianEM, EnsembleContactsLognormalEM

    if 'k2' in init_state.variables:
        xpdf = posterior.conditional_factory(k2=init_state.variables['k2'])
    else:
        xpdf = posterior
        
    structure_sampler = MyISD2FastHMCSampler(xpdf,
                                             init_state.variables['structures'],
                                             structures_timestep, structures_nsteps)
    subsamplers = OrderedDict(structures=structure_sampler)
    em = posterior.likelihoods['contacts'].error_model

    if 'k2' in init_state.variables:
        if isinstance(em, EnsembleContactsGaussianEM):
            from hicisd2.npsamplers import PrecisionSampler
            k2_sampler = PrecisionSampler(init_state=init_state.variables['k2'],
                                          variable_name='k2',
                                          adapt_stepsize=True, greater_zero=True)
        elif isinstance(em, EnsembleContactsLognormalEM):
            k2_sampler = LognormalGammaSampler(n_datapoints=len(posterior.likelihoods['contacts'].forward_model.data_points))
        k2_sampler.pdf = posterior.conditional_factory(structures=init_state.variables['structures'])
        subsamplers.update(k2=k2_sampler)
            
    return subsamplers


def make_subsamplers_norm(posterior, init_state,
                          structures_timestep=1e-4, structures_nsteps=100,
                          norm_stepsize=1e1):
    from collections import OrderedDict
    from hicisd2.error_models.ensemble_contacts import EnsembleContactsGaussianEM, EnsembleContactsLognormalEM, EnsembleContactsPoissonEM

    xpdf = posterior.conditional_factory(norm=init_state.variables['norm'])
    structure_sampler = MyISD2FastHMCSampler(xpdf,
                                             init_state.variables['structures'],
                                             structures_timestep, structures_nsteps)
    subsamplers = OrderedDict(structures=structure_sampler)
    em = posterior.likelihoods['contacts'].error_model

    if isinstance(em, EnsembleContactsGaussianEM):
        from hicisd2.npsamplers import NormSampler
        norm_sampler = NormSampler(init_state.variables['norm'], 'norm',
                                   norm_stepsize, adapt_stepsize=True,
                                   greater_zero=True)
    elif isinstance(em, EnsembleContactsPoissonEM):
        from hicisd2.npsamplers import PoissonNormSampler
        norm_sampler = PoissonNormSampler(init_state.variables['norm'], 'norm',
                                          norm_stepsize, adapt_stepsize=True,
                                          greater_zero=True)
    elif isinstance(em, EnsembleContactsLognormalEM):
        from hicisd2.npsamplers import LognormalNormSampler
        norm_sampler = LognormalNormSampler(init_state.variables['norm'], 'norm',
                                            norm_stepsize, adapt_stepsize=True,
                                            greater_zero=True)
    subsamplers.update(norm=norm_sampler)

    if 'k2' in init_state.variables:
        if isinstance(em, EnsembleContactsGaussianEM):
            from hicisd2.npsamplers import PrecisionSampler
            k2_sampler = PrecisionSampler(init_state=init_state.variables['k2'],
                                          variable_name='k2',
                                          adapt_stepsize=True, greater_zero=True)
        elif isinstance(em, EnsembleContactsLognormalEM):
            k2_sampler = LognormalGammaSampler(n_datapoints=len(posterior.likelihoods['contacts'].forward_model.data_points))
        k2_sampler.pdf = posterior.conditional_factory(structures=init_state.variables['structures'], norm=init_state.variables['norm'])
        subsamplers.update(k2=k2_sampler)
        
    return subsamplers

from rexfw.statistics.writers import ConsoleStatisticsWriter


def setup_default_re_master(n_replicas, sim_path, comm):

    from rexfw.remasters import ExchangeMaster
    from rexfw.statistics import Statistics, REStatistics
    from rexfw.statistics.writers import StandardConsoleREStatisticsWriter, StandardFileMCMCStatisticsWriter, StandardFileREStatisticsWriter, StandardFileREWorksStatisticsWriter, StandardConsoleMCMCStatisticsWriter, StandardConsoleMCMCStatisticsWriter
    from rexfw.convenience import create_standard_RE_params
    from rexfw.convenience.statistics import create_standard_averages, create_standard_works, create_standard_stepsizes, create_standard_heats

    replica_names = ['replica{}'.format(i) for i in range(1, n_replicas + 1)]
    params = create_standard_RE_params(n_replicas)
        
    from rexfw.statistics.averages import REAcceptanceRateAverage, MCMCAcceptanceRateAverage
    from rexfw.statistics.logged_quantities import SamplerStepsize
    
    local_pacc_avgs = [MCMCAcceptanceRateAverage(r, 'structures')
                       for r in replica_names]
    # local_pacc_avgs += [MCMCAcceptanceRateAverage(r, 'weights')
    #                     for r in replica_names]
    re_pacc_avgs = [REAcceptanceRateAverage(replica_names[i], replica_names[i+1]) 
                    for i in range(len(replica_names) - 1)]
    stepsizes = [SamplerStepsize(r, 'structures') for r in replica_names]
    # stepsizes += [SamplerStepsize(r, 'weights') for r in replica_names]
    # stepsizes += [SamplerStepsize(r, 'k2') for r in replica_names]
    works = create_standard_works(replica_names)
    heats = create_standard_heats(replica_names)
    stats_path = sim_path + 'statistics/'
    stats_writers = [StandardConsoleMCMCStatisticsWriter(['structures',
                                                          #'weights'
                                                          #'k2',
                                                          ],
                                                         ['acceptance rate',
                                                          'stepsize']),
                     StandardFileMCMCStatisticsWriter(stats_path + '/mcmc_stats.txt',
                                                      ['structures',
                                                       #'weights'
                                                       ],
                                                      ['acceptance rate', 'stepsize'])
                    ]
    stats = Statistics(elements=local_pacc_avgs + stepsizes, 
                       stats_writer=stats_writers)
    re_stats_writers = [StandardConsoleREStatisticsWriter(),
                        StandardFileREStatisticsWriter(stats_path + 're_stats.txt',
                                                       ['acceptance rate'])]
    works_path = sim_path + 'works/'
    works_writers = [StandardFileREWorksStatisticsWriter(works_path)]
    re_stats = REStatistics(elements=re_pacc_avgs,
                            work_elements=works, heat_elements=heats,
                            stats_writer=re_stats_writers,
                            works_writer=works_writers)
    
    master = ExchangeMaster('master0', replica_names, params, comm=comm, 
                            sampling_statistics=stats, swap_statistics=re_stats)

    return master


def kth_diag_indices(a, k):
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
