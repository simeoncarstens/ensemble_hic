'''
'''

from rexfw.replicas import Replica


class CompatibleReplica(Replica):

    def get_energy(self, state):

        return -self.pdf.log_prob(**state.variables)
