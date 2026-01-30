import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# === STTC ===

import numpy as np

def proportion_spikes_within_dt(spikes_A, spikes_B, delta_t):
    if len(spikes_A) == 0:
        return 0.0
    count = 0
    j = 0
    for t in spikes_A:
        while j < len(spikes_B) and spikes_B[j] < t - delta_t:
            j += 1
        if j < len(spikes_B) and abs(spikes_B[j] - t) <= delta_t:
            count += 1
    return count / len(spikes_A)


def proportion_time_covered(spikes, delta_t, T):
    if len(spikes) == 0:
        return 0.0
    intervals = np.clip(
        np.vstack((spikes - delta_t, spikes + delta_t)).T,
        0, T
    )
    intervals = intervals[np.argsort(intervals[:, 0])]
    
    covered = 0.0
    start, end = intervals[0]
    for s, e in intervals[1:]:
        if s <= end:
            end = max(end, e)
        else:
            covered += end - start
            start, end = s, e
    covered += end - start
    return covered / T

def sttc(spikes_A, spikes_B, delta_t, T):
    PA = proportion_spikes_within_dt(spikes_A, spikes_B, delta_t)
    PB = proportion_spikes_within_dt(spikes_B, spikes_A, delta_t)
    TA = proportion_time_covered(spikes_A, delta_t, T)
    TB = proportion_time_covered(spikes_B, delta_t, T)

    return 0.5 * (
        (PA - TB) / (1 - PA * TB + 1e-12) +
        (PB - TA) / (1 - PB * TA + 1e-12)
    )

def get_sttc_matrix(spike_times, delta_t, T):
    from joblib import Parallel, delayed

    pairs = [(i, j) for i in range(len(spike_times)) for j in range(i+1, len(spike_times))]

    results = Parallel(n_jobs=8)(
        delayed(sttc)(spike_times[i], spike_times[j], delta_t, T)
        for i, j in pairs)
    
    n = len(spike_times)
    STTC = np.zeros((n, n))

    k = 0
    for i in range(n):
        for j in range(i+1, n):
            STTC[i, j] = results[k]
            STTC[j, i] = results[k]  # symmetric
            k += 1

    return STTC

# === FC, FCD ===

def compute_time_window_fc_t(ts, window_length=20, overlap=19):
    """
    Function used to compute the Functional Connectivity for a time window of a matrix from a time series of neural activity.
    
    :param ts: time series data, shape (n_samples, n_regions)
    :param window_length: length of the sliding window
    :param overlap: overlap between consecutive windows
    """
    n_samples, n_regions = ts.shape
    #    if n_samples < n_regions:
    #        print('ts transposed')
    #        ts=ts.T
    #        n_samples, n_regions = ts.shape

    window_steps_size = window_length - overlap
    n_windows = int(np.floor((n_samples - window_length) / window_steps_size + 1))

    # upper triangle indices
    Isupdiag = np.triu_indices(n_regions, 1)

    # compute FC for each window
    FC_t = np.zeros((int(n_regions * (n_regions - 1) / 2), n_windows))
    for i in range(n_windows):
        FCtemp = np.corrcoef(ts[window_steps_size * i:window_length + window_steps_size * i, :].T)
        FCtemp = np.nan_to_num(FCtemp, nan=0)
        FC_t[:, i] = FCtemp[Isupdiag]

    return FC_t

def compute_fcd(ts, window_length=20, overlap=19):
    """
    Function used to compute the Functional Connectivity Dynamics (FCD) matrix from a time series of neural activity.
    
    :param ts: time series data, shape (n_samples, n_regions)
    :param window_length: length of the sliding window
    :param overlap: overlap between consecutive windows
    """
    FC_t = compute_time_window_fc_t(ts, window_length, overlap)

    # compute FCD by correlating the FCs with each other
    FCD = np.corrcoef(FC_t.T)
    FCD = np.nan_to_num(FCD, nan=0)
    #return(np.var(np.triu(FCD, k=1)))

    return FCD

# === CLUSTERING ===

def run_kmeans(X, k):
    assert X.shape[0] < X.shape[1]

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X_z)

    labels = KMeans(k).fit_predict(X_pca)

    return X_pca, labels

def silhouette_for_kmeans(X):
    assert X.shape[0] < X.shape[1]
    vmin = 2
    vmax=8
    scores = []
    for i in range(10):
        for k in range(vmin, vmax):
            X_pca, labels = run_kmeans(X, k)
            scores.append(silhouette_score(X_pca, labels))

    scores = np.array(scores).reshape(10,6)
    plt.plot(scores.mean(axis=0))
    plt.xticks(range(vmax-vmin), range(vmin, vmax))
