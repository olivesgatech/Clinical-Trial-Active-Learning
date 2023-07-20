import torch
import pandas as pd
import numpy as np
from myUtils.trainer_updated import Trainer_New
from query_strategies import RandomSampling, LeastConfidenceSampler, EntropySampler, \
    MarginSampler, BadgeSampler, PatientDiverseSampler, \
    PatientDiverseEntropySampler, PatientDiverseEntropyMacroSampler, PatientDiverseMarginSampler, \
    PatientDiverseLeastConfidenceSampler, PatientDiverseBadgeSampler, ClinicallyDiverseSampler, \
    ClinicallyDiverseEntropySampler, ClinicallyDiverseBadgeSampler, CoresetSampler

# Previous experiment; not utilized
def return_sampler(fixed, dynamic, visit, strategy, test_pool, start_idxs_test, inds, week_te_pool_ind):
    placeholder = np.array([None])
    #print('STRATEGY1: ', strategy)
    if dynamic == 0 and fixed == False:
        sampler_test = placeholder  # if using entire visit's images at each round for example
        # This gets handled later in the code by just implementing a random sampler
        return sampler_test

    if visit == 'None':
        if strategy == 'rand':
            if fixed == True:
                sampler_test = RandomSampling(test_pool, start_idxs_test, inds)
            elif dynamic > 0:
                sampler_test = RandomSampling(test_pool, start_idxs_test, inds)
        elif strategy == 'least_conf':
            if fixed == True:
                sampler_test = LeastConfidenceSampler(test_pool, start_idxs_test, inds)
            elif dynamic > 0:
                sampler_test = LeastConfidenceSampler(test_pool, start_idxs_test, inds)
        elif strategy == 'entropy':
            if fixed == True:
                sampler_test = EntropySampler(test_pool, start_idxs_test, inds)
            elif dynamic > 0:
                sampler_test = EntropySampler(test_pool, start_idxs_test, inds)
        elif strategy == 'margin':
            if fixed == True:
                sampler_test = MarginSampler(test_pool, start_idxs_test, inds)
            elif dynamic > 0:
                sampler_test = MarginSampler(test_pool, start_idxs_test, inds)
        elif strategy == 'badge':
            if fixed == True:
                sampler_test = BadgeSampler(test_pool, start_idxs_test)
            elif dynamic > 0:
                sampler_test = BadgeSampler(test_pool, start_idxs_test)
        elif strategy == 'coreset':
            if fixed == True:
                sampler_test = CoresetSampler(test_pool, start_idxs_test, inds)
            elif dynamic > 0:
                sampler_test = CoresetSampler(test_pool, start_idxs_test, inds)
        return sampler_test

    elif visit == 'yes': # CONTINUAL LEARNING
        print('STRATEGY: ', strategy)
        if strategy == 'rand':
            if fixed == True:
                sampler_test = RandomSampling(test_pool, start_idxs_test, placeholder)
            elif dynamic > 0:
                sampler_test = RandomSampling(test_pool, start_idxs_test, week_te_pool_ind)
        elif strategy == 'least_conf':
            if fixed == True:
                sampler_test = LeastConfidenceSampler(test_pool, start_idxs_test, placeholder)
            elif dynamic > 0:
                sampler_test = LeastConfidenceSampler(test_pool, start_idxs_test, week_te_pool_ind)
        elif strategy == 'entropy':
            if fixed == True:
                sampler_test = EntropySampler(test_pool, start_idxs_test, placeholder)
            elif dynamic > 0:
                sampler_test = EntropySampler(test_pool, start_idxs_test, week_te_pool_ind)
        elif strategy == 'margin':
            if fixed == True:
                sampler_test = MarginSampler(test_pool, start_idxs_test, placeholder)
            elif dynamic > 0:
                sampler_test = MarginSampler(test_pool, start_idxs_test, week_te_pool_ind)
        elif strategy == 'badge':
            if fixed == True:
                sampler_test = BadgeSampler(test_pool, start_idxs_test)
            elif dynamic > 0:
                sampler_test = BadgeSampler(test_pool, start_idxs_test)
        elif strategy == 'coreset':
            if fixed == True:
                sampler_test = CoresetSampler(test_pool, start_idxs_test, inds)
            elif dynamic > 0:
                sampler_test = CoresetSampler(test_pool, start_idxs_test, inds)
        return sampler_test

    return placeholder
