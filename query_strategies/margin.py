import numpy as np
from .sampler import Sampler


class MarginSampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, total):
        '''Constructor implemented in sampler'''
        super(MarginSampler, self).__init__(n_pool, start_idxs, total)

    def query(self, n, probs):
        '''Returns the samples with the smallest prediction margin between the two highess prediction values.
        Parameters:
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''
        # get probabilities and their indices
        indices = probs['indices']
        probabilities = probs['probs']

        # get smallest margins
        sorted_probs = np.sort(probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        prob_inds = np.argsort(margins)[:n]

        # derive final indices
        inds = indices[prob_inds]
        if inds.shape[1] >= 1:
            inds = inds.squeeze()

        return inds
