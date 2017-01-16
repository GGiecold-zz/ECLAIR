#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistics/Statistical_tests.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com; ggiecold@jimmy.harvard.edu


"""ECLAIR is a package for the robust and scalable 
inference of cell lineages from gene expression data.

ECLAIR achieves a higher level of confidence in the estimated lineages 
through the use of approximation algorithms for consensus clustering and by combining the information from an ensemble of minimum spanning trees 
so as to come up with an improved, aggregated lineage tree. 

In addition, the present package features several customized algorithms for assessing the similarity between weighted graphs or unrooted trees and for estimating the reproducibility of each edge to a given tree.

References
----------
* Giecold, G., Marco, E., Trippa, L. and Yuan, G.-C.,
"Robust Lineage Reconstruction from High-Dimensional Single-Cell Data". 
ArXiv preprint [q-bio.QM, stat.AP, stat.CO, stat.ML]: http://arxiv.org/abs/1601.02748

* Strehl, A. and Ghosh, J., "Cluster Ensembles - A Knowledge Reuse Framework
for Combining Multiple Partitions".
In: Journal of Machine Learning Research, 3, pp. 583-617. 2002

* Conte, D., Foggia, P., Sansone, C. and Vento, M., 
"Thirty Years of Graph Matching in Pattern Recognition". 
In: International Journal of Pattern Recognition and Artificial Intelligence, 
18, 3, pp. 265-298. 2004
"""


from __future__ import print_function

from . import Class_tree_edges, Gaussian_KDE_HDF5

from collections import defaultdict, OrderedDict
import functools
import inspect
from itertools import combinations, izip
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import psutil
from scipy.special import betainc, chdtrc
from scipy.stats import entropy, distributions, norm, rankdata
from sklearn.metrics import normalized_mutual_info_score
import subprocess
from sys import exit, maxint
import tables
from tempfile import NamedTemporaryFile
import time


__all__ = ['build_contingency_table', 'chi_squared_and_likelihood_ratio',
           'inter_clusterings_info_theory_measures',
           'pairwise_distances_correlations', 'robustness_metrics']


def memory():
    r"""Determine memory specifications of the machine.

    Returns
    -------
    mem_info : dictonary
        Holds the current values for the total, free and used memory of the system.
    """

    mem_info = dict()

    for k, v in psutil.virtual_memory().__dict__.iteritems():
           mem_info[k] = int(v)
           
    return mem_info


def get_chunk_size(N, n):
    """Given a two-dimensional array with a dimension of size 'N', 
        determine the number of rows or columns that can fit into memory.

    Parameters
    ----------
    N : int
        The size of one of the dimensions of a two-dimensional array.  

    n : int
        The number of arrays of size 'N' times 'chunk_size' that can fit in memory.

    Returns
    -------
    chunk_size : int
        The size of the dimension orthogonal to the one of size 'N'. 
    """

    mem_free = memory()['free']
    if mem_free > 60000000:
        chunk_size = int(((mem_free - 10000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 40000000:
        chunk_size = int(((mem_free - 7000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 14000000:
        chunk_size = int(((mem_free - 2000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 8000000:
        chunk_size = int(((mem_free - 1400000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 2000000:
        chunk_size = int(((mem_free - 900000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 1000000:
        chunk_size = int(((mem_free - 400000) * 1000) / (4 * n * N))
        return chunk_size
    else:
        print("\nERROR: ECLAIR: Statistical_tests: get_chunk_size: "
              "this machine does not have enough free memory resources "
              "to perform ensemble clustering.\n")
        exit(1)


def get_ranks(observations, number_of_duplicates):
    """For each entry in array 'observations', there are possibly many 
       value duplicates.
       The procedure herewith ensures that the value duplicates are assigned 
       a rank equal to the average of their positions 
       in the ascending order of the values. 
       This function avoids having to consider vectors of very large sizes
       and of many identical entries.
    """
    
    number_of_duplicates = np.array(number_of_duplicates, dtype = int, copy = False)
        
    assert np.all(number_of_duplicates > 0)
    # Please note that the situation where a value occurs 
    # in 'observations' at different entries 
    # and only a few of the corresponding entries 
    # in 'number_of_duplicates' have zero value is easily handled.
    # Yet we must ensure that all entries
    # in 'number_of_duplicates' are positive so as to avoid
    # the situation where a value in 'observations' has all zero entries
    # in 'number_of_duplicates'.
    
    temp_ranks = rankdata(observations, method = 'min')
    ranks = np.copy(temp_ranks)

    used_indices = np.zeros(0, dtype = int)
    all_indices = np.arange(0, ranks.size, dtype = int)

    max_rank = int(np.amax(ranks))
    for rank in xrange(1, max_rank + 1):
        ind = np.where(temp_ranks == rank)[0]
        used_indices = np.append(used_indices, ind)
        not_used_yet = np.setdiff1d(all_indices, used_indices, True)

        szum = number_of_duplicates[ind].sum()

        ranks[not_used_yet] += szum - 1
        ranks[ind] += (szum - 1) / 2.0

    return ranks


def build_contingency_track_calls(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args_name = list(OrderedDict.fromkeys(inspect.getargspec(f)[0] + kwargs.keys()))
        args_dict = OrderedDict(list(izip(args_name, args)) + list(kwargs.iteritems()))

        name = args_dict.get('hdf5_file_name', None)
        wrapper.hdf5_file_name = name

        return f(*args, **kwargs)

    wrapper.hdf5_file_name = None
    
    return wrapper


@build_contingency_track_calls
def build_contingency_table(hdf5_file_name, cluster_IDs_1, cluster_IDs_2, 
                            test_set = None):
    """Build a contingency table where each row corresponds to a pair of
        clusters in tree_1, each column to such a pair in tree_2 
        and the entries of this matrix report how many pairs of two samples 
        have been classified as belonging to such pairs of clusters 
        according to the classifier associated with tree_1 
        and the one associated with tree_2, respectively.

    Parameters
    ----------
    hdf5_file_name : string or file object

    cluster_IDs_1 :

    cluster_IDs_2 :

    test_set : None

    Returns
    -------
    cluster_IDs_1 :

    cluster_IDs_2 :
    """

    assert cluster_IDs_1.size == cluster_IDs_2.size

    if test_set is not None:
        test_set = np.array(test_set, dtype = int, copy = False)

    reject_ind_1 = np.where(cluster_IDs_1 == -1)[0]
    reject_ind_2 = np.where(cluster_IDs_2 == -1)[0]
    reject_ind = np.union1d(reject_ind_1, reject_ind_2)

    cluster_IDs_1 = np.delete(cluster_IDs_1, reject_ind)
    cluster_IDs_2 = np.delete(cluster_IDs_2, reject_ind)

    N_1 = np.amax(cluster_IDs_1) + 1
    unq_1 = np.arange(0, N_1)

    N_2 = np.amax(cluster_IDs_2) + 1
    unq_2 = np.arange(0, N_2)

    clusters_1_to_cells = defaultdict(list)

    if test_set is not None:
        temp = np.take(cluster_IDs_1, test_set)
    else:
        temp = cluster_IDs_1

    for c_1 in xrange(N_1):
        cells_in_c_1 = np.where(temp == c_1)[0]
        clusters_1_to_cells[int(c_1)].append(cells_in_c_1)    

    clusters_2_to_cells = defaultdict(list)

    if test_set is not None:
        temp = np.take(cluster_IDs_2, test_set)
    else:
        temp = cluster_IDs_2

    for c_2 in xrange(N_2):
        cells_in_c_2 = np.where(temp == c_2)[0]
        clusters_2_to_cells[int(c_2)].append(cells_in_c_2)

    fileh = tables.open_file(hdf5_file_name, 'w')

    atom = tables.Int64Atom()

    inter_trees_group = fileh.create_group(fileh.root, "inter_trees_group")

    contingency_table = fileh.create_carray(inter_trees_group, 'contingency_table', atom, (N_1 * (N_1 + 1) / 2, N_2 * (N_2 + 1) / 2), 'Each row corresponds to a pair of clusters in tree_1, each column to such a pair in tree_2, and the entries of this matrix report how many pairs of two samples have been classified as belonging to such pairs of clusters according to the classifier associated with tree_1 and the one associated with tree_2, respectively', filters = None)

    for i, clusters_1_pair in enumerate(combinations(unq_1, 2)):
        c_a, c_b = clusters_1_pair
        cells_in_a = clusters_1_to_cells[c_a][0]
        cells_in_b = clusters_1_to_cells[c_b][0]

        if (test_set is not None) and (cells_in_a.size == 0 or cells_in_b.size == 0):
            contingency_table[i] = 0
            continue

        for j, clusters_2_pair in enumerate(combinations(unq_2, 2)):
            c_d, c_e = clusters_2_pair
            cells_in_d = clusters_2_to_cells[c_d][0]
            cells_in_e = clusters_2_to_cells[c_e][0]

            if (test_set is not None) and (cells_in_d.size == 0 or cells_in_e.size == 0):
                contingency_table[i, j] = 0
                continue
                
            N_overlapping = 0

            n_d = np.intersect1d(cells_in_d, cells_in_a, assume_unique = True).size
            n_e = np.intersect1d(cells_in_e, cells_in_b, assume_unique = True).size
            N_overlapping += n_d * n_e

            n_d = np.intersect1d(cells_in_d, cells_in_b, assume_unique = True).size  
            n_e = np.intersect1d(cells_in_e, cells_in_a, assume_unique = True).size
            N_overlapping += n_d * n_e

            contingency_table[i, j] = N_overlapping

        for l in xrange(N_2 * (N_2 - 1) / 2, N_2 * (N_2 + 1) / 2):
            cells_2 = clusters_2_to_cells[l - (N_2 * (N_2 - 1) / 2)][0]

            if (test_set is not None) and cells_2.size == 0:
                contingency_table[i, l] = 0
                continue
 
            n_1_a = np.intersect1d(cells_in_a, cells_2, assume_unique = True).size
            n_1_b = np.intersect1d(cells_in_b, cells_2, assume_unique = True).size

            contingency_table[i, l] = n_1_a * n_1_b

    for k in xrange(N_1 * (N_1 - 1) / 2, N_1 * (N_1 + 1) / 2):
        cells_1 = clusters_1_to_cells[k - (N_1 * (N_1 - 1) / 2)][0]

        if (test_set is not None) and cells_1.size == 0:
                contingency_table[k] = 0
                continue

        for j, clusters_2_pair in enumerate(combinations(unq_2, 2)):
            c_d, c_e = clusters_2_pair
            cells_in_d = clusters_2_to_cells[c_d][0]
            cells_in_e = clusters_2_to_cells[c_e][0]

            if (test_set is not None) and (cells_in_d.size == 0 or cells_in_e.size == 0):
                contingency_table[k, j] = 0
                continue
                                       
            n_2_d = np.intersect1d(cells_in_d, cells_1, assume_unique = True).size
            n_2_e = np.intersect1d(cells_in_e, cells_1, assume_unique = True).size

            contingency_table[k, j] = n_2_d * n_2_e

        for l in xrange(N_2 * (N_2 - 1) / 2, N_2 * (N_2 + 1) / 2):
            cells_2 = clusters_2_to_cells[l - (N_2 * (N_2 - 1) / 2)][0]

            if (test_set is not None) and cells_2.size == 0:
                contingency_table[k, l] = 0
                continue

            n_12 = np.intersect1d(cells_1, cells_2, assume_unique = True).size 
            contingency_table[k, l] = n_12 * (n_12 - 1) / 2

    fileh.close()

    return cluster_IDs_1, cluster_IDs_2


def get_mean_and_variance(hdf5_file_name, axis, N_samples, dist_mat,
                          get_highest_entry = False):
    """

    Parameters
    ----------
    hdf5_file_name : string or file object

    N_samples : int
        Denotes the number of points in the dataset.
        Please note that the total number of samples used for
        computing most of the test statistics considered in the
        present module comprises the total number of pairs of such points,
        i.e. N_samples * (N_samples - 1) / 2.

    Returns
    -------
    E : float
   
    Var : float

    sum_vector : array of shape (contigency_table.shape[axis])

    highest_entry : int (optional, depending on 'get_highest_entry' at input)
    """

    assert axis in {0, 1}
    assert build_contingency_table.hdf5_file_name == hdf5_file_name

    fileh = tables.open_file(hdf5_file_name, 'r+')

    contingency_table = fileh.root.inter_trees_group.contingency_table

    E = 0.0
    Var = 0.0

    sum_check = 0
    sum_vector = np.zeros(0, dtype = int)

    if get_highest_entry:
        highest_entry = 0.0

    (n1, n2) = contingency_table.shape
    if axis:
        n2, n1 = n1, n2
    

    chunks_size = get_chunk_size(n2, 2)
    for i in xrange(0, n1, chunks_size):
        if axis:
            M = contingency_table[:, i:min(i+chunks_size, n1)]
        else: 
            M = contingency_table[i:min(i+chunks_size, n1)]

        if get_highest_entry:
            highest_entry = max(highest_entry, np.amax(M))

        szum = M.sum(axis = 0 if axis else 1)

        sum_check += szum.sum()
        sum_vector = np.append(sum_vector, szum)

        E_dot = np.dot(dist_mat[i:min(i+chunks_size, n1)], szum)
        Var_dot = np.dot(np.square(dist_mat[i:min(i+chunks_size, n1)]), szum)

        E += 2.0 * E_dot / N_samples
        Var += 2.0 * Var_dot / (N_samples - (2.0 / (N_samples - 1)))

    fileh.close()

    assert sum_check == N_samples * (N_samples - 1) / 2.0

    E /= float(N_samples - 1)
    Var /= float(N_samples - 1)
    Var -= (float((N_samples ** 2) - N_samples) / float((N_samples ** 2) - N_samples - 2)) * (E ** 2)

    if get_highest_entry:
        return E, Var, sum_vector, highest_entry
    
    return E, Var, sum_vector


def one_to_max(array_in):
    """Alter a vector of cluster labels to a dense mapping. 
       Given that this function is herein always called after passing 
       a vector to the function checkcl, one_to_max relies on the assumption 
       that cluster_run does not contain any NaN entries.

    Parameters
    ----------
    array_in : a list or one-dimensional array
        The list of cluster IDs to be processed.
    
    Returns
    -------
    result : one-dimensional array
        A massaged version of the input vector of cluster identities.
    """
    
    x = np.asanyarray(array_in)
    N_in = x.size
    array_in = x.reshape(N_in)    

    sorted_array = np.sort(array_in)
    sorting_indices = np.argsort(array_in)

    last = np.nan
    current_index = -1
    for i in xrange(N_in):
        if last != sorted_array[i] or np.isnan(last):
            last = sorted_array[i]
            current_index += 1

        sorted_array[i] = current_index

    result = np.empty(N_in, dtype = int)
    result[sorting_indices] = sorted_array

    return result


def pairwise_distances_correlations(hdf5_file_name, cluster_IDs_1, cluster_IDs_2,
                                    MST_dist_mat_1, MST_dist_mat_2, test_set = None,
                                    output_directory = '.', tag = '', 
                                    verbose = False, sluggish_alternative = False):
    """A very fast and scalable computation of Pearson's correlation
       coefficient and Spearman's rank correlation
       for a group of pairwise distances.

       Given a matrix of shortest paths along minimum-spanning tree_1, 
       given another such matrix for tree_2, alongside two vectors
       identifying where in the nodes of those trees the samples from
       a single dataset have been classified,
       for the set of all pairs of data-points, we can compute
       their distance along tree_1 and compare it with the
       distance by which their nodes are separated along tree_2.
       The present procedure computes the correlation between
       those two groups of distances.

       For 100,000 samples, there are 19,999,900,000 pairs to consider.
       It gets quadratically worse for larger datasets.
       A vector of that many elements exceeds the memory resources
       of most machines.
       
       My approach avoids dealing with that large a pair of 
       cells by taking advantage of the information provided
       by the nodes in which they belong in tree_1 and in tree_2.
       
       Essentially, we have consider the combination of all possible
       pairs of 2 nodes in tree_1 and all possible pairs in tree_2.
       With N_1 clusters in tree_1 and N_2 clusters in tree_2,
       our algorithm takes only of the order 
       N_1 * (N_1 + 1) * N_2 * (N_2 + 1) in memory resources
       or disk storage (making use of an HDF5 data structure).

       Even on a small number of points (about 440) clustered into two trees,
       each tree featuring around 10 nodes, this approach returns
       four times as fast as the naive implementation.

    Parameters
    ----------
    hdf5_file_name : string or file object

    cluster_IDs_1 : array, shape (n_samples, )
        A vector recording for each sample the identity
        of the cluster it has been assigned to 
        in the clustering associated with tree_1

    cluster_IDs_2 : array, shape (n_samples, )
        A vector of cluster labels for each sample,
        according to the clustering from which 
        minimum-spanning tree_2 originates.

    MST_dist_mat_1 : array of shape (n_clusters_1, n_clusters_1)
        A matrix recording the shortest route from each node
        of tree_1 to all other vertices, as computed
        for instance via Dijkstra's algorithm.

    MST_dist_mat_2 : array of shape (n_clusters_2, n_clusters_2)
        A matrix recording the shortest paths from each node
        of tree_2 to all other vertices, along the minimum spanning tree.

    test_set : array, optional (default = None)

    output_directory : file object or string
        Where to output the plots obtained as by-products
        of our procedure.

    tag : string, optional (default = '')
        An identifier to be appended to the scatter plot
        illustrating the relation between the distances along
        tree_1 and tree_2. 

    verbose : bool, optional (default = True)
        Controls the display of status messages to the standard output.

    sluggish_alternative : bool, optional (default = False)
        A flag determining whether to use our fast and scalable
        algorithm or the naive procedure for obtaining the same result.

    Returns
    -------
    pearson_rho_12 : float
        The pearson correlation coefficient between two variables.
        The first variable corresponds to the distances separating
        along a minimum spanning tree the nodes
        in which two samples have been classified, for each pair
        of samples.
        The second variable is computed the same way along a second
        minimum spanning tree.

    spearman_rho_12 : float
        The pearson correlation coefficient between two variables.
        The first variable corresponds to the distances separating
        along a minimum spanning tree the nodes
        in which two samples have been classified, for each pair
        of samples.
        The second variable is computed the same way along a second
        minimum spanning tree.
    """

    start_t = time.time()

    assert MST_dist_mat_1.shape[0] == MST_dist_mat_1.shape[1]
    assert MST_dist_mat_2.shape[0] == MST_dist_mat_2.shape[1]

    assert cluster_IDs_1.size == cluster_IDs_2.size

    if test_set is not None:
        test_set = np.array(test_set, dtype = int, copy = False)
        N_samples = test_set.size
    else:
        N_samples = cluster_IDs_1.size

    N_1 = np.amax(cluster_IDs_1) + 1
    assert N_1 == MST_dist_mat_1.shape[0]

    N_2 = np.amax(cluster_IDs_2) + 1
    assert N_2 == MST_dist_mat_2.shape[0]

    all_equal_flag = False
    if test_set is not None:
        reduced_cluster_IDs_1 = np.take(cluster_IDs_1, test_set)

        unq = np.unique(reduced_cluster_IDs_1)
        absent_clusters_1 = np.setdiff1d(np.arange(N_1), unq, True)

        reduced_cluster_IDs_1 = one_to_max(reduced_cluster_IDs_1)

        reduced_cluster_IDs_2 = np.take(cluster_IDs_2, test_set)

        unq = np.unique(reduced_cluster_IDs_2)
        absent_clusters_2 = np.setdiff1d(np.arange(N_2), unq, True)

        reduced_cluster_IDs_2 = one_to_max(reduced_cluster_IDs_2)

        reduced_MST_dist_mat_1 = np.delete(MST_dist_mat_1, absent_clusters_1, 0)
        reduced_MST_dist_mat_1 = np.delete(MST_dist_mat_1, absent_clusters_1, 1)

        reduced_MST_dist_mat_2 = np.delete(MST_dist_mat_2, absent_clusters_2, 0)
        reduced_MST_dist_mat_2 = np.delete(MST_dist_mat_2, absent_clusters_2, 1)

        if np.array_equal(reduced_cluster_IDs_1, reduced_cluster_IDs_2) and np.allclose(reduced_MST_dist_mat_1, reduced_MST_dist_mat_2):
            all_equal_flag = True

    else:
        if np.array_equal(cluster_IDs_1, cluster_IDs_2) and np.allclose(MST_dist_mat_1, MST_dist_mat_2):
            all_equal_flag = True
            
    if all_equal_flag:
        if verbose:
            print("\nECLAIR_robustness\t INFO\t:\n'pairwise_distances_correlations' "
                  "has identified as equal the two vectors of cluster IDs "
                  "and the two shortest-path matrices along the two trees "
                  "provided as input. A correlation of 1.0 has therefore been "
                  "returned; no scatter plot provided in this case.\n")

        return 1.0, 1.0

    # if we want to compare with the straightforward
    # but inefficient computation of the pearson correlation:
    if sluggish_alternative and N_samples < 10000:
        from scipy.stats import pearsonr, spearmanr

        def process_cell_pairs(N_samples, cluster_IDs_1, cluster_IDs_2, MST_dist_mat_1, MST_dist_mat_2):

            dist_1 = np.zeros(N_samples * (N_samples - 1) / 2, dtype = float)
            dist_2 = np.zeros(N_samples * (N_samples - 1) / 2, dtype = float)

            for i, cell_pair in enumerate(combinations(np.arange(N_samples), 2)):
                cell_i, cell_j = cell_pair

                cluster_a = cluster_IDs_1[cell_i]
                cluster_b = cluster_IDs_1[cell_j]

                d_ab = MST_dist_mat_1[cluster_a, cluster_b]
                dist_1[i] = d_ab

                cluster_c = cluster_IDs_2[cell_i]
                cluster_d = cluster_IDs_2[cell_j]

                d_cd = MST_dist_mat_2[cluster_c, cluster_d]
                dist_2[i] = d_cd

            return dist_1, dist_2

        if test_set is not None:
            dist_1, dist_2 = process_cell_pairs(N_samples, reduced_cluster_IDs_1, reduced_cluster_IDs_2, reduced_MST_dist_mat_1, reduced_MST_dist_mat_2)
        else:
            dist_1, dist_2 = process_cell_pairs(N_samples, cluster_IDs_1, cluster_IDs_2, MST_dist_mat_1, MST_dist_mat_2)

        pearson_rho_12, _ = pearsonr(dist_1, dist_2)
        pearson_rho_12 = round(pearson_rho_12, 4)

        confidence_interval = pearson_correlation_confidence_interval(pearson_rho_12, N_samples * (N_samples - 1) / 2.0, 0.05)

        spearman_rho_12, _ = spearmanr(dist_1, dist_2)
        spearman_rho_12 = round(spearman_rho_12, 4)

        fig = plt.figure()
        plt.xlabel('Tree 1', fontsize = 10)
        plt.ylabel('Tree 2', fontsize = 10)
        plt.axis([- 0.5, np.amax(MST_dist_mat_1 if test_set is None else reduced_MST_dist_mat_1) + 0.5, - 0.5, np.amax(MST_dist_mat_2 if test_set is None else reduced_MST_dist_mat_2) + 0.5])

        fig.suptitle("Scatter plot of the pairwise distances between cells,\ncompared between two minimum spanning trees.\nPearson's rho = {pearson_rho_12} and 95% confidence interval: ({confidence_interval[0]}; {confidence_interval[1]}); Spearman's rho = {spearman_rho_12}.".format(**locals()), fontsize = 10)

        plt.scatter(dist_1, dist_2, marker = 'o', alpha = 0.1, rasterized = True)

        dist_1_span = np.arange(np.amin(dist_1), np.amax(dist_1) + 1)

        slope = pearson_rho_12 * (np.std(dist_2) / np.std(dist_1))
        intercept = dist_2.mean() - (slope * dist_1.mean())

        plt.plot(dist_1_span, dist_1_span * slope + intercept)

        plt.savefig(output_directory + '/inter_trees_pairwise_cells_distances_{}.pdf'.format(tag))
        plt.close(fig)

        gaussian_kde_plot(dist_1, dist_2, output_directory, tag)

        end_t = time.time()

        if verbose:
            print("\nECLAIR_robustness\t INFO\t:\n'pairwise_distances_correlations' took {} seconds to compare the respective distances along two minimum spanning trees between each of {} pairs of samples in your dataset (ignoring points tagged as noise or disconnected).\n".format(round(end_t - start_t, 2), N_samples * (N_samples - 1) / 2))

        return pearson_rho_12, spearman_rho_12

        # END of sluggish, alternative computation. Used only to compare
        # with and illustrate the efficiency of the method expressed on the 
        # following lines of the present 'pairwise_distances_correlations'
        # procedure.

    assert build_contingency_table.hdf5_file_name == hdf5_file_name

    iu_1 = np.triu_indices(N_1, k = 1)
    dist_1 = MST_dist_mat_1[iu_1]
    dist_1 = np.append(dist_1, np.zeros(N_1, dtype = float))

    iu_2 = np.triu_indices(N_2, k = 1)
    dist_2 = MST_dist_mat_2[iu_2]
    dist_2 = np.append(dist_2, np.zeros(N_2, dtype = float))

    E_1, Var_1, sum_vector, highest_entry = get_mean_and_variance(hdf5_file_name, 0, N_samples, dist_1, True) 

    # The following 3 lines of code take into account the situation 
    # where only a test_set of samples are considered; 
    # they may very well be present in only some of the nodes of clustering 1.
    valid_rows = (sum_vector != 0) 
    sum_vector = np.compress(valid_rows, sum_vector)
    dist_1 = np.compress(valid_rows, dist_1)
 
    ranked_dist_1 = get_ranks(dist_1, sum_vector)

    E_2, Var_2, sum_vector = get_mean_and_variance(hdf5_file_name, 1, N_samples, dist_2)

    # The following 3 lines of code take into account the situation 
    # where only a test_set of samples are considered; 
    # they may very well be present in only some of the nodes of clustering 2.
    valid_cols = (sum_vector != 0)
    sum_vector = np.compress(valid_cols, sum_vector)
    dist_2 = np.compress(valid_cols, dist_2)

    ranked_dist_2 = get_ranks(dist_2, sum_vector)

    fileh = tables.open_file(hdf5_file_name, 'r+')

    contingency_table = fileh.root.inter_trees_group.contingency_table

    assert (N_1 * (N_1 + 1) / 2, N_2 * (N_2 + 1) / 2) == contingency_table.shape

    dist_1 -= E_1
    dist_2 -= E_2

    pearson_rho_12 = 0.0
    spearman_rho_12 = 0.0
    
    c = 0    
    chunks_size = get_chunk_size(N_2 * (N_2 + 1) / 2, 4)
    for i in xrange(0, N_1 * (N_1 + 1) / 2, chunks_size):
        max_ind = min(i + chunks_size, N_1 * (N_1 + 1) / 2)

        M = contingency_table[i:max_ind]
        M = np.compress(valid_rows[i:max_ind], M, axis = 0)
        M = np.compress(valid_cols, M, axis = 1)

        nrows = M.shape[0]

        x = np.dot(M, dist_2.reshape(-1, 1))
        szum = np.dot(dist_1[c:c+nrows], x)[0]
            
        pearson_rho_12 += 2.0 * szum / (N_samples - (2.0 / (N_samples - 1)))

        ranked_dist_1_chunk = ranked_dist_1[c:c+nrows].reshape(-1, 1)
        diff_ranks = np.tile(ranked_dist_1_chunk, dist_2.size)
        diff_ranks -= ranked_dist_2
        diff_ranks = np.square(diff_ranks)

        spearman_rho_12 -= 6.0 * np.sum(M * diff_ranks)
        spearman_rho_12 /= float(N_samples * (N_samples - 1) / 2)

        c += nrows

    pearson_rho_12 /= N_samples - 1
    pearson_rho_12 /= sqrt(Var_1) * sqrt(Var_2)
    pearson_rho_12 = round(pearson_rho_12, 4)

    confidence_interval = pearson_correlation_confidence_interval(pearson_rho_12, N_samples * (N_samples - 1) / 2.0, 0.05)

    spearman_rho_12 /= (float(N_samples * (N_samples - 1) / 2) ** 2) - 1
    spearman_rho_12 += 1
    spearman_rho_12 = round(spearman_rho_12, 4)

    dist_1 += E_1
    dist_2 += E_2

    fig = plt.figure()
    plt.xlabel('Tree 1', fontsize = 10)
    plt.ylabel('Tree 2', fontsize = 10)
    plt.axis([np.amin(dist_1) - 0.5, np.amax(dist_1) + 0.5, np.amin(dist_2) - 0.5, np.amax(dist_2) + 0.5])

    fig.suptitle("Scatter plot of the pairwise distances between cells,\ncompared between two minimum spanning trees.\nPearson's rho = {pearson_rho_12} and 95% confidence interval: ({confidence_interval[0]}; {confidence_interval[1]}); Spearman's rho = {spearman_rho_12}.".format(**locals()), fontsize = 10)

    with tables.open_file('kde_data_storage.h5', 'w') as tmp_fh:
        dataset = tmp_fh.create_earray(tmp_fh.root, 'dataset', tables.FloatAtom(), (2, 0), filters = None)
        repetitions = tmp_fh.create_earray(tmp_fh.root, 'repetitions', tables.Int64Atom(), (1, 0), filters = None)

        c = 0
        for i in xrange(N_1 * (N_1 + 1) / 2):
            contingency_table_i = contingency_table[i]
            contingency_table_i = np.compress(valid_cols, contingency_table_i)

            mask = (contingency_table_i != 0)

            if np.all(mask == False):
                continue

            Y = np.compress(mask, dist_2)
            X = np.full(Y.size, dist_1[c])

            Y += 0.5 * np.random.randn(Y.size)
            Y = np.clip(Y, 0, maxint)

            X += 0.5 * np.random.randn(X.size)
            X = np.clip(X, 0, maxint)

            dataset.append(np.vstack((X, Y)))
            repetitions.append(np.atleast_2d(np.compress(mask, contingency_table_i)))

            rgb_alpha_colors = np.zeros((Y.size, 4), dtype = float)
            rgb_alpha_colors[:, 2] = 1.0

            temp = np.divide(contingency_table_i, float(highest_entry))[mask]
            rgb_alpha_colors[:, 3] = temp

            plt.scatter(X, Y, marker = 'o', color = rgb_alpha_colors, rasterized = True)

            c += 1

    g_kde = Gaussian_KDE_HDF5.gaussian_kde('kde_data_storage.h5')

    Cov = g_kde.covariance_matrix
    means = g_kde.means

    slope = pearson_rho_12 * (sqrt(Cov[1,1]) / sqrt(Cov[0,0]))
    intercept = means[1] - (slope * means[0])

    dist_1_span = np.arange(np.amin(dist_1), np.amax(dist_1) + 1)
    plt.plot(dist_1_span, dist_1_span * slope + intercept)

    plt.savefig(output_directory + '/inter_trees_pairwise_cells_distances_{}.pdf'.format(tag))
    plt.close(fig)

    g_kde.plot(output_directory, tag, 250)

    subprocess.call(['rm', './kde_data_storage.h5'])

    end_t = time.time()

    if verbose:
        print("\nECLAIR_robustness\t INFO\t:\n'pairwise_distances_correlations' took {} seconds to compare the respective distances along two minimum spanning trees between each of {} pairs of samples in your dataset (ignoring points tagged as noise or disconnected).\n".format(round(end_t - start_t, 2), N_samples * (N_samples - 1) / 2))

    eensy = np.finfo(np.float32).eps

    if pearson_rho_12 == 1:
        pearson_rho_12 -= eensy

    if spearman_rho_12 == 1:
        spearman_rho_12 -= eensy

    return pearson_rho_12, spearman_rho_12


def gaussian_kde_plot(X, Y, output_directory = '.', tag = '', N_bins = 100):

    from scipy.stats import kde

    assert isinstance(N_bins, int) and N_bins > 1

    g_kde = kde.gaussian_kde(np.vstack((X, Y)))

    x_i, y_i = np.mgrid[np.amin(X):np.amax(X):N_bins*1j, np.amin(Y):np.amax(Y):N_bins*1j]
    z_i = g_kde(np.vstack((x_i.flatten(), y_i.flatten())))

    fig = plt.figure()
    fig.suptitle("Gaussian KDE of the pairwise distances between cells.", fontsize = 10)

    plt.pcolormesh(x_i, y_i, z_i.reshape(x_i.shape))

    fig.tight_layout()
    plt.savefig(output_directory + '/Gaussian_kde_inter_trees_pairwise_cells_distances_{}.pdf'.format(tag))
    plt.close(fig)


def chi_squared_and_likelihood_ratio(hdf5_file_name, median_dist_1, median_dist_2,
                                     test_set = None, output_directory = '.', 
                                     tag = '', yates_correction = True):
    """

    Parameters
    ----------
    hdf5_file_name : string or file object

    median_dist_1 : array of shape (n_clusters_1, n_clusters_1)
        For each pair of nodes of an ensemble clustering, 
        records the medians of the distributions of distances 
        over all runs of subsampling and clusterings underlying 
        this ensemble clustering.  

    median_dist_2 : array of shape (n_clusters_2, n_clusters_2)
        For each pair of nodes of a consensus clustering, 
        records the medians of the distributions of distances 
        over all runs of subsampling and clusterings out of which
        this consensus has been achieved.

    N_samples : int
        Denotes the number of points in the dataset.
        Please be advised that the total number of samples used for
        computing most of the test statistics considered in the
        present module comprises the total number of pairs of such points,
        i.e. N_samples * (N_samples - 1) / 2.

    test_set : array, optional (default = None)

    output_directory :

    tag :

    yates_correction :

    Returns
    -------
    chi_squared_p_value : float

    likelihood_ratio_p_value : float
    """

    assert build_contingency_table.hdf5_file_name == hdf5_file_name

    if test_set is not None:
        test_set = np.array(test_set, dtype = int, copy = False)

    (N_1, y_1) = median_dist_1.shape
    (N_2, y_2) = median_dist_2.shape

    assert N_1 == y_1
    assert N_2 == y_2

    del y_1
    del y_2

    iu_1 = np.triu_indices(N_1, k = 1)
    dist_1 = median_dist_1[iu_1]
    dist_1 = np.append(dist_1, np.zeros(N_1, dtype = int))

    iu_2 = np.triu_indices(N_2, k = 1)
    dist_2 = median_dist_2[iu_2]
    dist_2 = np.append(dist_2, np.zeros(N_2, dtype = int))

    fileh = tables.open_file(hdf5_file_name, 'r+')

    contingency_table = fileh.root.inter_trees_group.contingency_table
    assert (N_1 * (N_1 + 1) / 2, N_2 * (N_2 + 1) / 2) == contingency_table.shape

    def get_sum_vector(axis = 0):

        sum_vector = np.zeros(0, dtype = int)

        (n1, n2) = contingency_table.shape
        if axis:
            n2, n1 = n1, n2
    
        chunks_size = get_chunk_size(n2, 2)
        for i in xrange(0, n1, chunks_size):
            if axis:
                M = contingency_table[:, i:min(i+chunks_size, n1)]
            else: 
                M = contingency_table[i:min(i+chunks_size, n1)]

            szum = M.sum(axis = 0 if axis else 1)

            sum_vector = np.append(sum_vector, szum)

        return sum_vector

    sum_vector = get_sum_vector(0)

    N_samples = (1 + sqrt(1 + 8 * sum_vector.sum())) / 2

    if test_set is not None:
        assert N_samples == test_set.size

    valid_rows = (sum_vector != 0) 
    dist_1 = np.compress(valid_rows, dist_1)

    sum_vector = get_sum_vector(1)
    assert sum_vector.sum() == (N_samples * (N_samples - 1)) / 2

    valid_cols = (sum_vector != 0) 
    dist_2 = np.compress(valid_cols, dist_2)

    I = np.amax(dist_1) + 1
    J = np.amax(dist_2) + 1

    X_ij_table = np.zeros((I, J), dtype = np.int64)

    c = 0
    chunks_size = get_chunk_size(N_2 * (N_2 + 1) / 2, 4)
    for i in xrange(0, N_1 * (N_1 + 1) / 2, chunks_size):
        max_ind = min(i + chunks_size, N_1 * (N_1 + 1) / 2)

        M = contingency_table[i:max_ind]
        M = np.compress(valid_rows[i:max_ind], M, 0)
        M = np.compress(valid_cols, M, 1)

        nrows = M.shape[0]

        dist_1_chunk = dist_1[c:c+nrows]
        ixgrid = np.ix_(dist_1_chunk, dist_2)
   
        np.add.at(X_ij_table, ixgrid, M)
        # ensures that results are summed for elements that are indexed 
        # more than once

        c += nrows

    fileh.close()

    if np.any(X_ij_table < 0):
        raise ValueError('\nAll entries in the contingency table must be non-negative.\n')

    total_mass = X_ij_table.sum()
    total_mass = float(total_mass)

    diags_mass = np.diag(X_ij_table, 0).sum()
    diags_mass += np.diag(X_ij_table, 1).sum()
    diags_mass += np.diag(X_ij_table, -1).sum()

    fig = plt.figure()
    c = plt.pcolor(X_ij_table, cmap = plt.cm.Reds, edgecolor = 'k')
    plt.colorbar()
    plt.margins(0.3)
    plt.text(3, -1.3, 'Contingency table for the distances separating\neach pair of cells, according to tree 1 vs. tree 2\nFraction of mass along main three diagonals: {}'.format(round(diags_mass / total_mass, 4)), 
             fontsize = 6, horizontalalignment = 'center')
    plt.xticks(np.arange(J) + 0.5, [str(j) for j in np.arange(J)], fontsize = 6)
    plt.yticks(np.arange(I) + 0.5, [str(i) for i in np.arange(I)], fontsize = 6)
    plt.savefig(output_directory + '/inter_tree_median_distances_contingency_table_{}.pdf'.format(tag))
    plt.close(fig)

    X_is = X_ij_table.sum(axis = 1, keepdims = True)
    X_js = X_ij_table.sum(axis = 0, keepdims = True)

    n = N_samples * (N_samples - 1) / 2.0

    assert X_is.sum() == n
    assert X_js.sum() == n

    p_is = np.divide(X_is, n)
    p_js = np.divide(X_js, n)
    p_ij_table = np.dot(p_is, p_js)

    del X_is
    del X_js

    if np.any(p_ij_table == 0):
        raise ValueError('\nThe table of expected values computed from the contigency table stored in file {} and from the matrices of distances provided has at least a zero element.'.format(hdf5_file_name))

    dof = (I - 1) * (J - 1)

    if dof == 0:
        return (0.0, 0), (0.0, 0)
    # addresses the degenerate case when the contingency table is unidimensional
    # implying that the observations match exactly the expected frequencies
    # resulting in a null chi squared test statistic and likelihood ratio.

    if dof == 1 and yates_correction:
        temp = np.subtract(p_ij_table * n, X_ij_table)
        X_ij_table = np.add(X_ij_table, 0.5 * np.sign(temp))

    temp1 = np.divide(X_ij_table, n)

    temp2 = temp1 - p_ij_table
    temp2 = np.divide(temp2, p_ij_table)
    temp2 = np.square(temp2)
    temp2 = p_ij_table * temp2

    chi_squared = n * temp2.sum()
    chi_squared_p_value = chi_squared_prob(chi_squared, dof)

    temp2 = np.divide(temp1, p_ij_table)

    condition = np.ravel(temp1 != 0)
    temp2 = np.compress(condition, temp2)
    
    temp2 = np.log(temp2)

    temp2 = np.compress(condition, temp1) * temp2

    likelihood_ratio = 2.0 * n * temp2.sum()
    likelihood_ratio_p_value = chi_squared_prob(likelihood_ratio, dof)

    return chi_squared_p_value, likelihood_ratio_p_value


def chi_squared_prob(score, dof):

    assert score > 0

    return chdtrc(dof, score)


def incomplete_beta_integral(a, b, x):
    """Returns the incomplete beta function, namely:

    I_x(a,b) = 1/B(a,b)*(Integral(0,x) of t^(a-1)(1-t)^(b-1) dt),

    where a > 0, b > 0 and B(a,b) = G(a)*G(b)/(G(a+b)).
    G(a) denotes the gamma function of a.
    The standard broadcasting rules apply to a, b, and x.

    Parameters
    ----------
    a : array_like or float > 0
    b : array_like or float > 0
    x : array_like or float
        x will be made no greater than 1.0.

    Returns
    -------
    betai : ndarray
        The incomplete beta function for the parameters provided.
    """

    x = np.array(x, copy = True)
    x = np.where(x < 1.0, x, 1.0)  # values of x larger than 1.0 are clipped.

    return betainc(a, b, x)


def get_pearson_correlation_significance(pearson_rho, sample_size):

    assert isinstance(sample_size, int)
    assert sample_size > 0

    eensy = np.finfo(np.float32).eps
    if pearson_rho == 1.0:
        pearson_rho -= eensy

    dof = sample_size - 2

    if np.abs(pearson_rho) == 1.0:
        p_value = 0.0
    else:
        t_squared = pearson_rho * pearson_rho * dof
        t_squared /= (1.0 - pearson_rho) * (1.0 + pearson_rho)

        p_value = incomplete_beta_integral(0.5 * dof, 0.5, dof / (dof + t_squared))

    return p_value


def get_spearman_correlation_significance(spearman_rho, sample_size):

    assert isinstance(sample_size, int)
    assert sample_size > 0

    eensy = np.finfo(np.float32).eps
    if spearman_rho == 1.0:
        spearman_rho -= eensy

    dof = sample_size - 2

    olderr = np.seterr(divide='ignore')
    try:
        t = spearman_rho * sqrt(dof)
        t /= sqrt((spearman_rho + 1.0) * (1.0 - spearman_rho))
    finally:
        np.seterr(**olderr)

    p_value = distributions.t.sf(np.abs(t), dof) * 2

    return p_value


def pearson_correlation_confidence_interval(rho, sample_size, alpha):
    """Computes an approximate 1 - 'alpha' confidence interval for a given
       Pearson's correlation coefficient and a given sample size,
       according to Fisher's method.
    """

    assert 0 < alpha < 1

    eensy = np.finfo(np.float32).eps
    if rho == 1.0:
        rho -= eensy

    theta = np.arctanh(rho)

    z = norm.interval(1 - alpha)[1]
    half_length = z / sqrt(sample_size - 3)

    a, b = theta - half_length, theta + half_length

    return round((np.exp(2*a) - 1) / (np.exp(2*a)+1), 5), round((np.exp(2*b)-1) / (np.exp(2*b)+1), 5)
       

def correct_for_multiple_testing(p_values, correction_method = 'Benjamini-Hochberg'):       
                                                         
    assert correction_method in {'Bonferroni', 'Benjamini-Hochberg', 'Holm-Bonferroni'}

    p_values = np.asfarray(p_values)
    m = p_values.size

    if reduce(operator.mul, p_values.shape, 1) != max(p_values.shape):
        p_values = p_values.reshape(m)
    
    if correction_method == 'Bonferroni':
        new_p_values = m * p_values
    elif correction_method == 'Benjamini-Hochberg':
        ranks = rankdata(p_values, method = 'ordinal')

        inverse_ranks = np.reciprocal(ranks.astype(float))
        C_m = inverse_ranks.sum()

        new_p_values = np.multiply(p_values, C_m * m * inverse_ranks) 
    elif correction_method == 'Holm-Bonferroni':
        ranks = rankdata(p_values, method = 'ordinal')
        new_p_values = p_values * np.subtract(m + 1, ranks)

    return new_p_values


def reject_hypotheses(corrected_p_values, alpha, statistical_test,
                      multiple_testing_method):

    N_rejected_hypotheses = np.where(corrected_p_values < alpha)[0].size

    print('\nUsing the {multiple_testing_method} method for multiple testing of {corrected_p_values.size} hypothesis tests under a null hypothesis of statistical independence for the {statistical_test}, {N_rejected_hypotheses} hypotheses have been rejected while ensuring that the overall probability of falsely rejecting any null hypothesis is less than or equal to {alpha}\n'.format(**locals()))


def get_jensen_shannon(cluster_IDs_1, cluster_IDs_2):

    # checking that all entries are integers:
    assert not np.any(np.mod(cluster_IDs_1, 1))
    assert not np.any(np.mod(cluster_IDs_2, 1))

    unk1, cnt1 = np.unique(cluster_IDs_1, return_counts = True)
    unk2, cnt2 = np.unique(cluster_IDs_2, return_counts = True)

    joint_max = max(unk1[-1], unk2[-1])

    P = np.zeros(joint_max + 1, dtype = float)
    Q = np.zeros(joint_max + 1, dtype = float)

    P[unk1] = cnt1
    Q[unk2] = cnt2

    M = 0.5 * (P + Q)

    JSD = 0.5 * (entropy(P, M) + entropy(Q, M))

    return JSD


def info_theory_heatmap(scores_matrix, mean_score, output_directory, 
                        measure_flag = 'Jensen-Shannon'):

    assert measure_flag in {'Jensen-Shannon', 'mutual information'}

    N_trees = scores_matrix.shape[0]
    mean_score = round(mean_score, 4)

    fig = plt.figure()

    c = plt.pcolor(scores_matrix, cmap = plt.cm.Reds, edgecolor = 'k')
    c.update_scalarmappable()
    ax = c.get_axes()
    for p in c.get_paths():
        x, y = p.vertices[:-2, :].mean(0)
        if x == y:
            ax.text(x, y, '%s' % 'N/A', ha = 'center', va = 'center')
    
    plt.colorbar()

    plt.margins(0.3)
    plt.text(3, -1.3, 'Heatmap of pairwise {measure_flag} scores\nbetween distinct clusterings.\nAverage {measure_flag} score between\ndistinct clusterings = {mean_score}'.format(**locals()), 
             fontsize = 6, horizontalalignment = 'center')

    rows = list('C_{}'.format(i) for i in xrange(N_trees))
    columns = rows
    plt.xticks(np.arange(N_trees) + 0.5, columns, fontsize = 6, rotation = 'vertical')
    plt.yticks(np.arange(N_trees) + 0.5, rows, fontsize = 6)

    if measure_flag == 'Jensen-Shannon':
        plt.savefig(output_directory + '/inter_trees_jensen_shannon.pdf')
    elif measure_flag == 'mutual information':
        plt.savefig(output_directory + '/inter_trees_mutual_info.pdf')

    plt.close(fig)


def inter_clusterings_info_theory_measures(input_directory, name_tags,
                                           output_directory = '.', 
                                           test_set_flag = False):
    """Given a set of minimum spanning trees stemming from as many different
       clusterings, this routine computes for each pair of trees
       their normalized mutual information score and Jensen-Shannon metric
       (it has been proven that the square root of the Jensen-Shannon divergence
        is indeed a metric).
       Heatmap are output for convenient visualization.

    Parameters
    ----------
    input_directory : file object or string
        A directory holding the folders recording for each tree
        its adjacency matrix, a matrix of shortest paths 
        between each of its vertices, etc.
   
    name_tags : list
        The identifiers to the directories within 'input_directory' 
        where information on the structure of each tree is stored.

    output_directory : file object or string, 
                       optional (default = '.', the current directory)
        Where to output and store the heatmap featuring
        all normalized mutual information scores between all
        pairs of distinct clusterings and their associated trees.

    test_set_flag : bool, optional (default = False)

    Returns
    -------
    mutual_info_mean : float
        The average of the mutual information scores over all 
        distinct pairs of vectors of cluster identities.    

    mutual_info_std : float
        The standard deviation of the distribution of normalized
        mutual information scores over all distinct tuples
        of clustering vectors.

    jensen_shannon_mean : float
        The average of the Jensen-Shannon distances over all 
        distinct pairs of vectors of consensus cluster labels.

    jensen_shannon_std : float
        The standard deviation of the distribution of Jensen-Shannon
        distances over all distinct pairs of consensus clustering labels.
    """

    N_trees = len(name_tags)

    mutual_info_matrix = np.zeros((N_trees, N_trees), dtype = float) 
    # we're not considering the self-mutual information of a vector
    # of clustering IDs, which would be normalized to 1.0

    mutual_info_mean = 0.0
    mutual_info_std = 0.0
    
    jensen_shannon_matrix = np.zeros((N_trees, N_trees), dtype = float)
    # we're not considering the Jensen-Shannon distance of a vector
    # of clustering IDs with itself, which would be null anyway.

    jensen_shannon_mean = 0.0
    jensen_shannon_std = 0.0

    for i in xrange(N_trees - 1):
        with open(input_directory + '/'+ str(name_tags[i]) + '/consensus_labels.txt', 'r') as f:
            full_cluster_IDs_i = np.loadtxt(f, dtype = int)
  
        with open(output_directory + '/training_{}.txt'.format(i+1), 'r') as f:
            training_i = np.loadtxt(f, dtype = int)

        if i == 0:
            N_samples = full_cluster_IDs_i.size
        else:
            assert N_samples == full_cluster_IDs_i.size

        for j in xrange(i + 1, N_trees):
            with open(input_directory + '/' + str(name_tags[j]) + '/consensus_labels.txt', 'r') as f:
                cluster_IDs_j = np.loadtxt(f, dtype = int)

            with open(output_directory + '/training_{}.txt'.format(j+1), 'r') as f:
                training_j = np.loadtxt(f, dtype = int)

            if test_set_flag:
                training_sets = np.union1d(training_i, training_j)
                test_set = np.setdiff1d(np.arange(N_samples), training_sets, True)

                cluster_IDs_i = np.take(full_cluster_IDs_i, test_set)
                cluster_IDs_j = np.take(cluster_IDs_j, test_set)
            else:
                cluster_IDs_i = full_cluster_IDs_i

            q = normalized_mutual_info_score(cluster_IDs_i, cluster_IDs_j)
            mutual_info_matrix[i, j] = q
            mutual_info_matrix[j, i] = q

            mutual_info_mean += 2.0 * q / N_trees
            mutual_info_std += 2 * (q ** 2)
            mutual_info_std /= (N_trees - (2.0 / (N_trees - 1)))

            jensen_shannon_div = get_jensen_shannon(cluster_IDs_i, cluster_IDs_j)
            jensen_shannon_dist = sqrt(jensen_shannon_div)
            jensen_shannon_matrix[i, j] = jensen_shannon_dist
            jensen_shannon_matrix[j, i] = jensen_shannon_dist

            jensen_shannon_mean += 2.0 * jensen_shannon_dist / N_trees
            jensen_shannon_std += 2 * (jensen_shannon_dist ** 2)
            jensen_shannon_std /= (N_trees - (2.0 / (N_trees - 1)))

    mutual_info_mean /= N_trees - 1
    mutual_info_std /= N_trees - 1

    mutual_info_std -= mutual_info_mean ** 2
    mutual_info_std = sqrt(abs(mutual_info_std))

    jensen_shannon_mean /= N_trees - 1
    jensen_shannon_std /= N_trees - 1

    jensen_shannon_std -= jensen_shannon_mean ** 2
    jensen_shannon_std = sqrt(abs(jensen_shannon_std))

    info_theory_heatmap(mutual_info_matrix, mutual_info_mean, output_directory,
                        measure_flag = 'mutual information')

    info_theory_heatmap(jensen_shannon_matrix, jensen_shannon_mean, 
                        output_directory)

    return mutual_info_mean, mutual_info_std, jensen_shannon_mean, jensen_shannon_std


def get_graph_data(directory):

    with open(directory + '/consensus_labels.txt', 'r') as f:
        cluster_IDs = np.loadtxt(f, dtype = int)

    with open(directory + '/ensemble_distances_means.txt', 'r') as f:
        graph_means = np.loadtxt(f, dtype = float)

    with open(directory + '/ensemble_distances_medians.txt', 'r') as f:
        graph_medians = np.loadtxt(f, dtype = int)

    with open(directory + '/ensemble_distances_variances.txt', 'r') as f:
        graph_std = np.loadtxt(f, dtype = float)

    graph_std = np.sqrt(graph_std)

    return cluster_IDs, graph_means, graph_medians, graph_std


def get_MST_data(directory):

    with open(directory + '/consensus_labels.txt', 'r') as f:
        cluster_IDs = np.loadtxt(f, dtype = int)

    #with open(directory+ '/consensus_distances_matrix.txt', 'r') as f:
        #MST_means = np.loadtxt(f, dtype = float)

    with open(directory+ '/consensus_topological_distances_matrix.txt', 'r') as f:
        MST_medians = np.loadtxt(f, dtype = float)

    MST_medians = MST_medians.astype(int)

    return cluster_IDs, MST_medians, MST_medians, None
    # after several discussions and changing requirements,
    # we decided in the end that the comparison across ensemble trees
    # would concern only with the number of hops from one node to another.
    # I am leaving open the possibility of a future analysis
    # involving the means of the distribution over all the minimum spanning trees
    # underlying an ensemble tree.
    # The present function returns two pointers to the same array,
    # along with a null pointer in order to make use of the function
    # 'robustness_metrics' below. This function was designed with
    # a comparison of ensemble graphs in mind (see 'get_graph_data' above).


def robustness_metrics(max_N_clusters, input_directory, name_tags, output_directory,
                       test_set_flag = True, MST_flag = True):
    """
    
    Parameters
    ----------
    max_N_clusters : int

    input_directory : file object or string
   
    name_tags : list

    output_directory : file object or string

    test_set_flag : bool, optional (default = False)

    MST_flag : bool, optional (default = True)
    """

    if MST_flag:
        mutual_info_mean, mutual_info_std, jensen_shannon_mean, jensen_shannon_std = inter_clusterings_info_theory_measures(input_directory, name_tags, output_directory, test_set_flag)

    assert isinstance(max_N_clusters, int) and max_N_clusters > 1

    get_data = lambda directory: get_MST_data(directory) if MST_flag else get_graph_data(directory)

    N_trees = len(name_tags)

    pearson_rho_list = []
    spearman_rho_list = []

    pearson_rho_p_values = []
    spearman_rho_p_values = []
    chi_squared_p_values = []
    likelihood_ratio_p_values = []

    folder_name = '/inter_trees_plots' if MST_flag else '/inter_graphs_plots'
    
    try:
        os.makedirs(output_directory + folder_name)
    except EnvironmentError:
        if not os.path.isdir(output_directory + folder_name):
            print('\nrobustness_metrics\t ERROR\n')
            raise

    if MST_flag:
        ref_tree = name_tags[0]
        ref_ind = 0
            
        print('\nECLAIR_robustness\t INFO\t:\nreference tree used for tree edges distributions has its information stored in {}.\n'.format(ref_tree))

    for i in xrange(N_trees - 1):
        dir_i = input_directory + '/' + str(name_tags[i])
        cluster_IDs_i, dist_means_i, dist_medians_i, dist_std_i = get_data(dir_i)

        if test_set_flag:
            with open(output_directory + '/training_{}.txt'.format(i+1), 'r') as f:
                training_i = np.loadtxt(f, dtype = int)

        with Class_tree_edges.tree_edges(dir_i, np.amax(cluster_IDs_i), max_N_clusters - 1) as tree_edges_obj:

            for j in xrange(i + 1, N_trees):
                dir_j = input_directory + '/' + str(name_tags[j])
                cluster_IDs_j, dist_means_j, dist_medians_j, dist_std_j = get_data(dir_j)

                assert cluster_IDs_i.size == cluster_IDs_j.size             

                if test_set_flag:
                    with open(output_directory + '/training_{}.txt'.format(j+1), 'r') as f:
                        training_j = np.loadtxt(f, dtype = int)

                    training_sets = np.union1d(training_i, training_j)
                    test_set = np.setdiff1d(np.arange(cluster_IDs_j.size), training_sets, assume_unique = True)
                    N_samples = test_set.size
                else:
                    test_set = None
                    N_samples = cluster_IDs_i.size

                with NamedTemporaryFile('w', suffix = '.h5', delete = True, dir = './') as f:
                    cluster_IDs_i, cluster_IDs_j = build_contingency_table(f.name,
                                            cluster_IDs_i, cluster_IDs_j, test_set)

                    pearson_rho, spearman_rho = pairwise_distances_correlations(f.name, cluster_IDs_i, cluster_IDs_j, dist_means_i, dist_means_j, test_set, output_directory + folder_name, tag = '{}_{}'.format(i, j))

                    pearson_rho_list.append(pearson_rho)
                    spearman_rho_list.append(spearman_rho)

                    p_value = get_pearson_correlation_significance(pearson_rho,
                                                N_samples * (N_samples - 1) / 2)
                    pearson_rho_p_values.append(p_value)

                    p_value = get_spearman_correlation_significance(spearman_rho,
                                                  N_samples * (N_samples - 1) / 2)
                    spearman_rho_p_values.append(p_value)

                    chi_squared_p_value, likelihood_ratio_p_value = chi_squared_and_likelihood_ratio(f.name, dist_medians_i, dist_medians_j, test_set, output_directory + folder_name, tag = '{}_{}'.format(i, j))

                    chi_squared_p_values.append(chi_squared_p_value)
                    likelihood_ratio_p_values.append(likelihood_ratio_p_value)

                    if test_set_flag and MST_flag:
                        tree_edges_obj.update_tree_edges_distributions(dir_j, f.name)

                        tree_edges_obj.plot(output_directory + folder_name, 
                                            '{}_{}'.format(i, j))

                        tree_edges_obj.wipe_storage()

                    if (not test_set_flag) and MST_flag and (ref_tree == name_tags[i]):
                        tree_edges_obj.update_tree_edges_distributions(dir_j, f.name)

            if (not test_set_flag) and MST_flag and (ref_tree == name_tags[i]):
                tree_edges_obj.plot(output_directory + folder_name, ref_tree)

    pearson_rho_mean = round(np.mean(pearson_rho_list), 4)
    pearson_rho_std = round(np.std(pearson_rho_list), 4)

    spearman_rho_mean = round(np.mean(spearman_rho_list), 4)
    spearman_rho_std = round(np.std(spearman_rho_list), 4)

    for method in {'Bonferroni', 'Benjamini-Hochberg', 'Holm-Bonferroni'}:
        corrected_p_values = correct_for_multiple_testing(pearson_rho_p_values,
                                                     correction_method = method)
        reject_hypotheses(corrected_p_values, 0.05, "Pearson's correlation", method)
    
        corrected_p_values = correct_for_multiple_testing(spearman_rho_p_values,
                                                      correction_method = method)
        reject_hypotheses(corrected_p_values, 0.05, 
            "Spearman's rank order correlation", method)

        corrected_p_values = correct_for_multiple_testing(chi_squared_p_values,
                                                     correction_method = method)
        reject_hypotheses(corrected_p_values, 0.05, 
            "Pearson's chi squared coefficient", method)

        corrected_p_values = correct_for_multiple_testing(likelihood_ratio_p_values,
                                                          correction_method = method)
        reject_hypotheses(corrected_p_values, 0.05, 
            "likelihood ratio test statistics", method)

    if MST_flag:
        print('\nECLAIR_robustness\t INFO\t:\nthe average normalized mutual information between distinct pairs of consensus clusterings is {}, with a standard deviation of {}.\n'.format(round(mutual_info_mean, 4), round(mutual_info_std, 4)))

        print('\nECLAIR_robustness\t INFO\t:\nthe average Jensen-Shannon distance between distinct pairs of consensus clusterings is {}, with a standard deviation of {}.\n'.format(round(jensen_shannon_mean, 4), round(jensen_shannon_std, 4)))

    print("\nECLAIR_robustness\t INFO\t:\nfor each pair of {0}, if we compute the correlation between the distances between the nodes in which each pair of samples belong, we find an average of {pearson_rho_mean} for this set of Pearson correlation coefficients, with a standard deviation of {pearson_rho_std}.\nAs for Spearman's rank correlation coefficients, they average to {spearman_rho_mean}, with a standard deviation of {spearman_rho_std}.\n".format('trees' if MST_flag else 'ensemble graphs', **locals()))

    return pearson_rho_list

