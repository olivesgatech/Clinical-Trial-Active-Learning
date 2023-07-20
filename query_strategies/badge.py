import numpy as np
import pdb
from scipy import stats
from sklearn.metrics import pairwise_distances
from .sampler import Sampler


# kmeans ++ initialization from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
def init_centers(X, K, debug=False):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('Starting K-Means++')
    if debug:
        print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if debug:
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


class BadgeSampler(Sampler):
    '''Class for sampling the highest gradnorm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, ideal_path=None):
        '''Constructor implemented in sampler'''
        super(BadgeSampler, self).__init__(n_pool, start_idxs, total_given=None)

        # load embeddings if ideal sampling
        if ideal_path is not None:
            self.embeddings = np.load(ideal_path)

    def query(self, n, embeddings=None):
        '''Performs badge sampling with k-means++ for increased diversity.
        Parameters:
            :param embeddings: datastructure containing the gradient embeddings of the penultimate layer
            :type embeddings: dict
            :param n: number of samples to be queried
            :type n: int'''
        if embeddings is not None:
            # get probabilities and their indices
            indices = embeddings['indices']
            grad_embedding = embeddings['embeddings']
        else:
            indices = np.arange(self.embeddings.shape[0])
            indices = np.delete(indices, self.idx_current)
            grad_embedding = np.delete(self.embeddings, self.idx_current, axis=0)

        # get smallest margins
        embed_inds = init_centers(grad_embedding, n)

        # derive final indices
        inds = indices[embed_inds]

        return inds.squeeze()
