from isd2.samplers.hmc import ISD2FastHMCSampler, HMCSampleStats

class HMCSampler(ISD2FastHMCSampler):

    def get_last_draw_stats(self):
        return HMCSampleStats(self.last_move_accepted, self._nmoves, self.timestep)
