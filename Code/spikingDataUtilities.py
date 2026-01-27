'''
Utility functions to handle spiking data

SPDX-License-Identifier: GPL-3.0-or-later
Copyright © 2026 by Pietro Bozzo, Gabriele Casagrande, Gianmarco Cafaro, Giacomo Preti
'''

from scipy.io import loadmat
import numpy as np

def loadMATData(session):
    # load data from a .mat file
    #
    # arguments:
    #     session    string, path to .mat file, which must contain a structure with one double array per field
    #
    # output:
    #     out        dictionary

    data = loadmat(session)['data']
    out = {field: data[field][0,0] for field in data.dtype.names}
    out['protocol_names'] = ['sleep1','task1','sleep2','task2','sleep3'] # hard code names to avoid bug with string arrays

    return out

def restrict(samples,intervals,s_ind=False):
    # keep only samples falling in a set of intervals
    #
    # arguments:
    #     samples      (n,m) float, every row is [time stamp, value1, value2, …]
    #     intervals    (l,2) float, every row is [start time, stop time] for an interval
    #     s_ind        bool = False, if True, return also indices of kept samples in original samples
    #
    # output:
    #     samples      (p,m) float, restricted samples, i. e., samples[:,0] fall into intervals
    #     indices      (n) bool, optional, indicese of original samples which were kept

    is_ok = np.full((samples.shape[0]),False)
    for interval in intervals:
        is_ok = is_ok | ((samples[:,0] > interval[0]) & (samples[:,0] < interval[1]))

    if s_ind:
        return samples[is_ok], is_ok
    return samples[is_ok]