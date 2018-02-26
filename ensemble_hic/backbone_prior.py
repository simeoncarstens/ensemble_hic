import numpy as np, os, sys

from csb.statistics.pdf.parameterized import Parameter

from isd2.pdf.priors import AbstractPrior

from ensemble_hic.backbone_prior_c import backbone_prior_gradient


class BackbonePrior(AbstractPrior):

    def __init__(self, name, lower_limits, upper_limits, k_bb, n_structures,
                 mol_ranges):

        from isd2 import ArrayParameter
        from csb.statistics.pdf.parameterized import Parameter

        super(BackbonePrior, self).__init__(name)

        self.n_structures = n_structures
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self._register('k_bb')
        self['k_bb'] = Parameter(k_bb, 'k_bb')

        self._register_variable('structures', differentiable=True)
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

        self._mol_ranges = mol_ranges

    def _single_structure_log_prob(self, structure, ll, ul):

        x = structure.reshape(-1, 3)
        k_bb = self['k_bb'].value

        d = np.sqrt(np.sum((x[1:] - x[:-1]) ** 2, 1))
        
        u_viols = d > ul
        l_viols = d < ll
        delta = ul - ll

        return -0.5 * k_bb * (  np.sum((d[u_viols] - ul[u_viols]) ** 2 ) \
                              + np.sum((ll[l_viols] - d[l_viols]) ** 2))
    
    def _single_structure_gradient(self, structure, ll, ul):
        
        return backbone_prior_gradient(structure.ravel(), ll, ul, self['k_bb'].value)

    def _evaluate_log_prob(self, structures):

        log_prob = self._single_structure_log_prob
        X = structures.reshape(self.n_structures, -1, 3)
        mr = self._mol_ranges
        ll, ul = self.lower_limits, self.upper_limits

        def ss_lp(x):
            return np.sum([log_prob(x[mr[i]:mr[i+1]], ll[i], ul[i])
                           for i in range(len(mr) - 1)])
            
        return np.sum(map(lambda x: ss_lp(x), X))

    def _evaluate_gradient(self, structures):

        grad = self._single_structure_gradient
        X = structures.reshape(self.n_structures, -1, 3)
        mr = self._mol_ranges
        ll, ul = self.lower_limits, self.upper_limits

        def ss_grad(x):
            return np.concatenate([grad(x[mr[i]:mr[i+1]], ll[i], ul[i])
                                   for i in range(len(mr) - 1)])

        return np.concatenate(map(lambda x: ss_grad(x), X))
		
    def clone(self):

        copy = self.__class__(self.name,
                              self.lower_limits,
                              self.upper_limits,
                              self['k_bb'].value, 
                              self.n_structures,
                              self._mol_ranges)

        copy.fix_variables(**{p: self[p].value for p in self.parameters
                              if not p in copy.parameters})

        return copy


if __name__ == '__main__':

    from ensemble_hic.setup_functions import make_elongated_structures

    np.random.seed(32)

    n_beads = 60
    n_structures = 3
    n_mols = 3
    bead_radii = np.random.uniform(low=1,high=2,size=n_beads)
    mr = np.array([0,20,30,60])
    ul = np.array([bead_radii[mr[i]+1:mr[i+1]] + bead_radii[mr[i]:mr[i+1]-1]
                   for i in range(n_mols)])
    ll = np.array([np.zeros(mr[i+1]-mr[i]-1) for i in range(n_mols)])

    P = MultiMolBackbonePrior('asdfasdf', ll, ul, 28.0, 3, mr)
    X = [make_elongated_structures(bead_radii[mr[i]:mr[i+1]], n_structures).reshape(n_structures, -1, 3)
         for i in range(n_mols)]
    X = map(list, map(None, *X)) ## arcane way to transpose a list of uneven lists
    X = np.concatenate([np.concatenate(x) for x in X])
    X[n_beads:2*n_beads,1] += 8*3
    X[2*n_beads:,1] += 16*3


    if True:
        from stuff import numgrad
        g = P.gradient(structures=X.ravel())
        ng = numgrad(X.ravel(), lambda x: -P.log_prob(structures=X))
        print np.max(np.fabs(g-ng))

    from csb.statistics.samplers.mc.propagators import HMCPropagator, MDPropagator
    from csb.statistics.samplers import State

    class MyPDF(object):
        def log_prob(self, x):
            return P.log_prob(structures=x)
        def gradient(self, x, t=0.0):
            return P.gradient(structures=x)

    pdf = MyPDF()
    prop = HMCPropagator(pdf=pdf, gradient=pdf.gradient,
                         timestep=0.15, nsteps=50)
    prop = MDPropagator(gradient=pdf.gradient, timestep=0.01)
    traj = prop.generate(State(X.ravel(), np.random.normal(size=X.ravel().shape)),
                         500, True)

    from ensemble_hic.analysis_functions import write_ensemble
    X = np.array([x.position.reshape(n_beads * n_structures, 3).reshape(n_structures,-1,3) for x in traj])
    write_ensemble(X.reshape(-1, n_beads, 3), '/tmp/out.pdb')
