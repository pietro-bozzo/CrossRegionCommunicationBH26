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

    try:
        samples = np.array(samples)
    except Exception as e:
        raise e
    
    is_ok = np.full((samples.shape[0]),False)
    for interval in intervals:
        if len(interval) < 2:
            raise(ValueError('intervals must be a (n,2) array'))
        is_ok = is_ok | ((samples[:,0] > interval[0]) & (samples[:,0] < interval[1]))

    if s_ind:
        return samples[is_ok], is_ok
    return samples[is_ok]


def consolidateIntervals(intervals):
    # remove overlaps in a set of intervals, yielding its most compact description (the union of its elements)
    # e.g., [[1,4],[2,6]] will become [1,6]
    #
    # arguments:
    #     intervals    (:,2) float, every row is [start time, stop time] for an interval

    try:
        intervals = np.array(intervals)
    except Exception as e:
        raise e
    # promote 1d arrays to 2d
    if len(intervals.shape) == 1:
        intervals = intervals.reshape((1,-1))
    if intervals.shape[1] != 2:
        raise(ValueError('intervals must be a (n,2) array'))
    if (intervals[:,0] > intervals[:,1]).any():
        raise(ValueError('rows of intervals must be increasing'))

    # sort by start time
    intervals = intervals[intervals[:,0].argsort()]

    # flatten and argsort to find overlaps
    intervals = intervals.flatten()
    ind = intervals.argsort()

    # remove all ind which are followed by at least one smaller element
    m = ind[-2:].min()
    is_ok = [True,ind[-2] < ind[-1]]
    for i in range(3,ind.shape[0]+1):
        is_ok.append(ind[-i] < m)
        m = min(ind[-i],m)
    is_ok.reverse()
    ind = ind[is_ok]

    # remove consecutive odd elements
    is_odd = (ind % 2).astype(bool)
    ind = ind[np.concatenate((~is_odd[:-1] | ~is_odd[1:],[True]))]

    # rebuild intervals
    intervals = intervals[np.reshape(ind,(ind.shape[0]//2,2))]

    return intervals


def intersectIntervals(intervals):
    # intersect interval sets, obtaining the set of intervals which are contained in all inputs
    #
    # arguments:
    #     intervals       (n) list of (:,2), every element is a set of [start time, stop time] intervals
    #
    # output:
    #     intersection    (:,2) float, intersection between elements of input

    # 1: more than 2 intervals, recursive call
    if len(intervals) > 2:
        intervals = [intervals[0],intersectIntervals(intervals[1:])]

    # 2: intersect 2 interval sets

    # consolidate every interval set
    intervals = [consolidateIntervals(i) for i in intervals]

    # flatten both sets
    a = intervals[0].flatten()
    b = intervals[1].flatten()

    # ind[i] is odd iff a[i] falls in an interval of b
    ind = np.digitize(a,b)

    # handle case when a ∋ [10,20] and b ∋ [20,35] : interval [20,20] must not be in result
    end_nz = ((ind != 0) & (np.arange(ind.shape[0]) % 2)).astype(bool) # end_nz[i] is 1 iff ind[i] is not 0 and i is odd
    change_ind = a[end_nz] == b[ind[end_nz]-1] # change_ind[j] is 1 iff corresponding interval must be shortened
    find_end_nz = np.where(end_nz)[0]
    ind[find_end_nz[change_ind]] -= 1

    # if ind[2*i-1], ind[2*i] are equal and even, it's an interval of a to exclude entirely
    keep_ind = ((ind[::2] % 2).astype(bool) | (ind[::2] != ind[1::2])).repeat(2)
    ind = ind[keep_ind]
    a = a[keep_ind]

    # odd_ind[i] is index of an odd element of ind, which will be replaced by a[odd_ind[i]]
    odd_ind = np.where(ind % 2)[0]
    # round start to lower even number and stop to lower odd number
    start = ind[::2] - ind[::2] % 2
    stop = ind[1::2] -1 + ind[1::2] % 2
    # expand each start[i], stop[i] pair to include intervals in between
    new_ind = np.array([np.arange(start[i],stop[i]+1) for i in range(start.shape[0])]).flatten().astype(int)
    # remap odd_ind to point elements of new_ind, which will have to be drawn from a
    remapping = np.array([np.ones(start.shape),stop-start]).flatten('F').cumsum() - 1# CHECK IF 'F' IS OK
    new_odd_ind = remapping[odd_ind].astype(int) # remapping CONTAINS MATLAB style indeces!!

    # initialize intervals as [b[new_ind[i]],b[new_ind[i+1]]]
    intersection = b[new_ind]
    # when ind[j] was odd, replace corresponding element with a[odd_ind[j]]
    intersection[new_odd_ind] = a[odd_ind]
    # return as interval set
    intersection = intersection.reshape((intersection.shape[0]//2,2))

    return intersection


def spikeTimes(data, protocol_times, protocol_based=True):
    """
    Process spike data into rasters per protocol.
    Input:
    data: np.ndarray of shape (M, 2) with spike times and neuron indices (M is total spikes)
    protocol_times: List of (start, end) tuples for each protocol
    protocol_based: If True, split rasters by protocol intervals
    
    Output:
    protocol_rasters (protocol, neuron_id, spike_times): 
        List of length P (number of protocols), each element is a list of length N (number of neurons),
        each element is a list of spike times for that neuron during that protocol.
        When protocol_based is False, returns a single raster for all spikes.
    """
    protocol_rasters = []
    if data.size == 0:
        return protocol_rasters
    
    min_idx = int(np.min(data[:,1]))
    max_idx = int(np.max(data[:,1]))
    N = int(max_idx - min_idx + 1)

    # Precompute zero-based integer indices and times
    idxs = data[:,1].astype(int) - min_idx
    times = data[:,0]

    if not protocol_based:
        # If not protocol based, return all spikes as a single raster
        single_raster = [times[idxs == i].tolist() for i in range(N)]
        return [single_raster]

    for start, end in protocol_times:
        # Select spikes in interval 
        mask = (times >= start) & (times < end)
        sub_times = times[mask]
        sub_idxs = idxs[mask]

        # Gather spikes per neuron for this protocol
        single_raster = [sub_times[sub_idxs == i].tolist() for i in range(N)]
        protocol_rasters.append(single_raster)
        
    return protocol_rasters