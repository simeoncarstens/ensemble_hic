"""
Forward models back-calculating interaction counts / frequencies from
structure ensembles
"""
import numpy, sys, os

from csb.statistics.pdf.parameterized import Parameter

from binf import ArrayParameter
from binf.model.forwardmodels import AbstractForwardModel

from .forward_models_c import ensemble_contacts_evaluate
	

class EnsembleContactsFWM(AbstractForwardModel):

    def __init__(self, name, n_structures, contact_distances, data_points):
        """
        Forward model back-calculating contact frequencies by summing
        up single-structure contact matrices and scaling them with 
        a number.

        :param name: a unique name for this object, usually something like 
                     'ensemble_contacts_fwm'
        :type name: str

        :param n_structures: number of ensemble members
        :type n_structures: int

        :param contact_distances: contact distances for each data point
        :type contact_distances: :class:`numpy.ndarray`

        :param data_points: list of pairwise bead indices and corresponding
                            contact counts / frequencies.
                            shape: (# data points, 3). Second axis is
                            (1st bead index, 2nd bead index, count)
        :type data_points: :class:`numpy.ndarray`
        """
        super(EnsembleContactsFWM, self).__init__(name)

        self.data_points = data_points

        self.n_structures = n_structures

        self._register('contact_distances')
        self['contact_distances'] = ArrayParameter(contact_distances,
                                                   'contact_distance')

        self._register_variable('structures', differentiable=True)
        self._register_variable('smooth_steepness')
        self._register_variable('norm')
	    
        self.update_var_param_types(structures=ArrayParameter,
                                    smooth_steepness=Parameter, 
                                    norm=Parameter)
        self._set_original_variables()

    def _evaluate(self, structures, smooth_steepness, norm):
        """
        Evaluates the forward model, i.e., back-calculates contact
        data from a structure ensemble and other (nuisance) parameters

        :param structures: coordinates of structure ensemble
        :type structures: :class:`numpy.ndarray`

        :param smooth_steepness: determines the steepness of the smoothed
                                 contact function
        :type smooth_steepness: float

        :param norm: scaling parameter with which the sum
                     of single-structure contact matrices
                     is multiplied
        :type norm: float

        :returns: back-calculated contact frequency data
        :rtype: :class:`numpy.ndarray`
        """
        X = structures.reshape(self.n_structures, -1, 3)

        return ensemble_contacts_evaluate(X,
                                          norm,
                                          self['contact_distances'].value, 
                                          smooth_steepness,
                                          data_points=numpy.array(self.data_points,
                                                                  dtype=int))
    
    def clone(self):

        copy = self.__class__(name=self.name, n_structures=self.n_structures, 
                              contact_distances=self['contact_distances'].value, 
                              data_points=self.data_points)

        self._set_parameters(copy)
        
        return copy
		
    def _evaluate_jacobi_matrix(self, structures, smooth_steepness, norm):
        """
        In theory, this evaluates the Jacobian matrix of the forward model,
        but I usually hardcode the multiplication of this with the error
        model gradient in Cython (see :module:`.likelihoods_c`)
        """        
        raise NotImplementedError("Use fast likelihood gradients in " +
                                  "likelihoods_c.pyx instead!")    
