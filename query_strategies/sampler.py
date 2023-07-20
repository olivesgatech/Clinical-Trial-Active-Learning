import numpy as np

class Sampler:
    def __init__(self, n_pool, start_idxs, total_given):
        # init idx list containing elements in AL pool
        self.idx_current = np.arange(n_pool)[start_idxs]

        # init list of total elements mapped to binary variables
        self.total_pool = np.zeros(n_pool, dtype=int)
        self.total_pool[self.idx_current] = 1

        self.total_given = total_given

    def query(self, n, visit):
        '''Pure virtual query function. Content implemented by other submodules
        Parameters:
            :param n: number of samples to be queried
            :type n: int'''
        pass

    def query_te(self, n, visit):
        pass

    def update(self, new_idx, cont, add=np.array([])):
        '''Updates the current data pool with the newly queried idxs.
        Parameters:
            :param new_idx: idxs used for update
            :type new_idx: ndarray'''
        if cont == False:
            # active learning; retrain model at each round with all queried samples
            self.idx_current = np.append(self.idx_current, new_idx)
        elif len(add) == 0:
            # continual learning; only add newly queried samples to pool
            self.idx_current = new_idx
        else:
            # continual learning where we add some past samples to current pool [randomly]
            self.idx_current = np.append(add, new_idx)
        self.total_pool[new_idx] = 1
