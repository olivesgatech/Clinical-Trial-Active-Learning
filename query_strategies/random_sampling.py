import numpy as np
from .sampler import Sampler


class RandomSampling(Sampler):
    '''Class for random sampling algorithm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, total):
        super(RandomSampling, self).__init__(n_pool, start_idxs, total)

    def query(self, n, visit, opt=False):
        '''Performs random query of points'''
        previous_idxs = self.idx_current
        if visit.all() != None:
            # previous weeks samples (unused)
            if not opt:
                xx = np.isin(self.total_given, previous_idxs, invert=True)
                prev_unused_idxs = np.where(xx)[0]

                # For dynamic test set size where full visit is added at each round
                if len(prev_unused_idxs) == 0 or n == 0:
                    inds = visit
                    return inds[np.random.permutation(len(inds))] # randomize order

                prev_unused = self.total_given[prev_unused_idxs]
                # new allowed samples
                inds = visit # new visit's indexes
                inds = np.concatenate((prev_unused, inds))
                self.total_given = np.concatenate((self.total_given, visit))
            else:
                # only allow current visit
                inds = visit
        else:
            print('Retrospective random sampling')
            inds = np.where(self.total_pool == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]
