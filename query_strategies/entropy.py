import numpy as np
from .sampler import Sampler


class EntropySampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, total):
        '''Constructor implemented in sampler'''
        super(EntropySampler, self).__init__(n_pool, start_idxs, total)

    def query(self, n, probs):
        '''Returns samples with highest entropy in the output distribution.
        Parameters:
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''
        # get probabilities and their indices
        indices = probs['indices']
        probabilities = probs['probs']

        # get max entropy
        logs = np.log2(probabilities)
        mult = logs*probabilities
        entropy = np.sum(mult, axis=1)
        prob_inds = np.argsort(entropy)[:n]

        # derive final indices
        inds = indices[prob_inds]

        if inds.shape[1] >= 1:
            inds = inds.squeeze()

        return inds
