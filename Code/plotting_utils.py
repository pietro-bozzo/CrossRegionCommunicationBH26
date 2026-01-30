import matplotlib.pyplot as plt 
import numpy as np

def plot_fcd_with_states(labels, fcd):
    """
    Plotting the dynamical functional connectivity matrix (fcd) next to the sequence of clustered FC states.  
    
    :param labels: labels list from the kmeans clustering
    :param fcd: fcd matrix 
    """

    # Creating the pile of labels to be colored the same way 
    label_img = np.tile(labels[:, None], (1, len(labels)))
    label_img.shape == (len(labels), len(labels))

    plt.subplot(121)
    plt.imshow(fcd, cmap="coolwarm", origin="lower")
    plt.subplot(122)
    # Overlay: labels
    plt.imshow(label_img, cmap="tab10", origin="lower", alpha=1,
            interpolation="nearest")
    
def get_global_fc(FC_t, n_neurons):
    global_vec = FC_t.mean(axis=1)

    global_FC = np.zeros((n_neurons, n_neurons))
    iu = np.triu_indices(n_neurons, k=1)
    global_FC[iu] = global_vec
    global_FC = global_FC + global_FC.T
    np.fill_diagonal(global_FC, 0)

    return global_FC


def plot_clustered_fc(FC_t, n_neurons, labels):
    """
    Plots the clustered FC averaging the fc matrices that have the same label
    
    :param FC_t: FC matrices for all the time windows in the FCD used for the clustering
    :param n_neurons: number of neurons in the FC matrices
    :param n_clusters: number of clusters used in Kmeans
    :param lables: labels from the clustering 
    """

    global_FC = get_global_fc(FC_t, n_neurons)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    fig, axs = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))

    if n_clusters == 1:
        axs = [axs]

    iu = np.triu_indices(n_neurons, k=1)

    for i, lab in enumerate(unique_labels):

        fc_indices = np.where(labels == lab)[0] # indices of FCs belonging to this cluster
        vecs = FC_t[:, fc_indices]
        mean_vec = vecs.mean(axis=1) 

        # reconstruct FC matrix
        FC = np.zeros((n_neurons, n_neurons))
        FC[iu] = mean_vec
        FC = FC + FC.T
        np.fill_diagonal(FC, 1.0)

        diff_FC = FC - global_FC
        v = np.percentile(np.abs(diff_FC), 90)
        thr = np.percentile(np.abs(diff_FC), 90)
        masked = np.where(np.abs(diff_FC) >= thr, diff_FC, 0)

        axs[i].imshow(masked, cmap="coolwarm", vmin=-thr, vmax=thr)

        #im = axs[i].imshow(diff_FC, vmin=-v, vmax=v, cmap="coolwarm")
        axs[i].set_title(f"Cluster {lab}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    plt.show()

def plot_fcd_with_statesand_clusters(fcd, protocol_states, cluster_labels):

    state_to_int = {s: i for i, s in enumerate(['nrem', 'wake', 'rem'])}

    state_int = np.array([state_to_int[s] for s in protocol_states])
    state_img = np.tile(state_int[:, None], (1, len(state_int)))
    state_img.shape == (len(state_int), len(state_int))

    label_img = np.tile(cluster_labels[:, None], (1, len(cluster_labels)))
    label_img.shape == (len(cluster_labels), len(cluster_labels))

    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    # Max number of categories across both
    n_colors = max(len(np.unique(state_int)), len(np.unique(cluster_labels)))

    cmap = ListedColormap(plt.cm.tab10.colors[:n_colors])
    norm = BoundaryNorm(np.arange(-0.5, n_colors + 0.5), n_colors)

    sleep_handles = [
    mpatches.Patch(color=cmap(state_to_int[s]), label=s)
    for s in ['wake', 'nrem', 'rem']
    ]

    unique_cluster_labels = np.unique(cluster_labels)
    cluster_handles = [
        mpatches.Patch(color=cmap(i), label=f'Cluster {i}')
        for i in unique_cluster_labels
    ]

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(fcd, cmap="coolwarm", origin="lower")
    plt.title("FCD")

    plt.subplot(132)
    plt.imshow(
        state_img,
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest"
    )
    plt.title("Sleep state")
    plt.legend(handles=sleep_handles, loc="upper right", fontsize=8)

    plt.subplot(133)
    plt.imshow(
        label_img,
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest"
    )
    plt.title("FC cluster")
    plt.legend(handles=cluster_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
