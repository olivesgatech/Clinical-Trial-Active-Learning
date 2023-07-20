import numpy as np
from sklearn.metrics import pairwise_distances
from .sampler import Sampler
from .config import BaseConfig


# implementation inspired from https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
# as well as https://github.com/svdesai/coreset-al/blob/master/coreset.py
def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)
    idxs = []

    for i in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
    return idxs


class CoresetSampler(Sampler):
    def __init__(self, n_pool, start_idxs, total):
        super(CoresetSampler, self).__init__(n_pool, start_idxs, total)
        self.min_distances = None
        self.total_features = None
        self.already_selected = []

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:
            x = self.total_features[centers]  # pick only centers
            dist = pairwise_distances(self.total_features, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
        return

    def query(self, n: int, trainer):
        # initially updating the distances
        # idx_current is an array of all INDICES of currently selected images
        embeddings_un, unlabeled_indices = trainer.embeddings(mode='tr', loader_type='unlabeled')
        embeddings_lab, labeled_indices = trainer.embeddings(mode='tr', loader_type='labeled')
        self.total_features = np.concatenate((embeddings_lab, embeddings_un)) # n_data_samples x 1000
        total_indices = np.concatenate((labeled_indices, unlabeled_indices))

        labeled_inds_reset = np.arange(0, embeddings_lab.shape[0])
        self.already_selected = labeled_inds_reset

        self.update_dist(labeled_inds_reset, only_new=False, reset_dist=True)

        new_batch = []
        for _ in range(n):
            ind = np.argmax(self.min_distances)
            # if not self.already_selected:
            #     ind = np.random.choice(np.arange(self.dset_size))
            # else:
            #     ind = np.argmax(self.min_distances)
            assert ind not in self.already_selected
            self.update_dist([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        new_batch = np.array(new_batch)
        actual_idxs = total_indices[new_batch]
        return np.array(actual_idxs).astype(int)


    def query_te(self, n, trainer):
        embeddings_un = trainer.get_embeddings(mode='te', loader_type='unlabeled')
        embeddings_lab = trainer.get_embeddings(mode='te', loader_type='labeled')

        unlabeled_indices = embeddings_un['indices']

        # do coreset algorithm
        chosen = furthest_first(embeddings_un['nongrad_embeddings'], embeddings_lab['nongrad_embeddings'], n)

        # derive final indices
        inds = unlabeled_indices[chosen]
        return inds
