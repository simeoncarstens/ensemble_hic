'''
Compatible replica classes derived from :ref:`rexfw.replicas.Replica`.
The get_energy method has to be implemented such that
:ref:`binf.samplers.ISDState` objects can be dealt with.
'''

from rexfw.replicas import Replica


class CompatibleReplica(Replica):

    def get_energy(self, state):

        return -self.pdf.log_prob(**state.variables)
