"""
Structural priors restraining structures to be located within a sphere
to simulate, e.g., a nuclear wall
"""
import numpy

from csb.statistics.pdf.parameterized import Parameter

from binf import ArrayParameter
from binf.pdf.priors import AbstractPrior

from .sphere_prior_c import sphere_prior_gradient

class SpherePrior(AbstractPrior):

    def __init__(self, name, sphere_radius, sphere_k, n_structures,
                 bead_radii, sphere_center=None):
        """
        Structural Boltzmann-like prior distribution harmonically restraining
        all beads to be located within a sphere of a given radius

        :param name: a unique name for this object, usually 'sphere_prior'
        :type name: string

        :param sphere_radius: the radius of the sphere within which to
                              restrain the beads
        :type sphere_radius: float

        :param sphere_k: force constant
        :type sphere_k: float
    
        :param n_structures: number of ensemble members
        :type n_structures: int

        :param bead_radii: bead radii for each bead
        :type bead_radii: :class:`numpy.ndarray`

        :param sphere_center: coordinates of the sphere center,
                              if none, (0, 0, 0) is assumed
        :type sphere_center: :class:`numpy.ndarray`

        :returns: set-up spherical prior distribution object
        :rtype: :class:`.SpherePrior`    
        """
        super(SpherePrior, self).__init__(name)

        self.n_structures = n_structures
        self.bead_radii = bead_radii
        self.bead_radii2 = bead_radii ** 2

        self._register_variable('structures', differentiable=True)
        self._register('sphere_radius')
        self['sphere_radius'] = Parameter(sphere_radius, 'sphere_radius')
        self._register('sphere_k')
        self['sphere_k'] = Parameter(sphere_k, 'sphere_k')
        self._register('sphere_center')
        sphere_center = numpy.zeros(3) if sphere_center is None else sphere_center
        self['sphere_center'] = ArrayParameter(sphere_center, 'sphere_center')
        self.update_var_param_types(structures=ArrayParameter)
        self._set_original_variables()

    def _single_structure_log_prob(self, structure):

        r = self['sphere_radius'].value
        k = self['sphere_k'].value
        br = self.bead_radii
        X = structure.reshape(-1, 3)
        norms = numpy.sqrt(numpy.sum((X - self['sphere_center'].value[None,:])
        **2, 1))
        violating = norms + br > r
        
        return -0.5 * k * numpy.sum((norms[violating] + br[violating] - r) ** 2)

    def _single_structure_gradient(self, structure):

        X = structure.reshape(-1, 3)
        return sphere_prior_gradient(X,
                                     self['sphere_center'].value,
                                     self['sphere_radius'].value,
                                     self['sphere_k'].value,
                                     numpy.arange(len(X)),
                                     self.bead_radii,
                                     self.bead_radii2)

    def _evaluate_log_prob(self, structures):

        log_prob = self._single_structure_log_prob
        X = structures.reshape(self.n_structures, -1, 3)
		
        return numpy.sum(map(lambda x: log_prob(structure=x), X))

    def _evaluate_gradient(self, structures):

        grad = self._single_structure_gradient
        X = structures.reshape(self.n_structures, -1, 3)

        return numpy.concatenate(map(lambda x: grad(structure=x), X))
        
    def clone(self):

        copy = self.__class__(name=self.name,
                              sphere_radius=self['sphere_radius'].value,
                              sphere_k=self['sphere_k'].value,
                              n_structures=self.n_structures,
                              bead_radii=self.bead_radii,
                              sphere_center=self['sphere_center'].value)

        copy.set_fixed_variables_from_pdf(self)

        return copy
