#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Build_instance/ECLAIR_core.py


# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


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

try:
    input = raw_input
except NameError:
    pass

import Concurrent_AP as AP            
import Cluster_Ensembles as CE        
import DBSCAN_multiplex               
import Density_Sampling               

from collections import defaultdict, namedtuple
import datetime    
import igraph     
from math import floor, sqrt
import numpy as np
import os
import psutil
import random
import scipy.sparse
from scipy.spatial.distance import _validate_vector
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
import subprocess
from sys import exit
import tables
import time


__all__ = ['tree_path_integrals', 'ECLAIR_processing']


Data_info = namedtuple('Data_info', "data_file_name expected_N_samples "
                       "skip_rows cell_IDs_column extra_excluded_columns "
                       "time_info_column")
AP_parameters = namedtuple('AP_parameters', "clustering_method max_iter "
                           "convergence_iter")
DBSCAN_parameters = namedtuple('DBSCAN_parameters', "clustering_method minPts "
                               "eps quantile metric")
HIERARCHICAL_parameters = namedtuple('HIERARCHICAL_parameters', 
                                     'clustering_method k')
KMEANS_parameters = namedtuple('KMEANS_parameters', 'clustering_method k')
CC_parameters = namedtuple('CC_parameters', 'N_runs sampling_fraction N_cc')
Holder = namedtuple('Holder', "N_samples subsample_size N_runs name_tag "
                    "method run error_count")


def memory():
    """Determine memory specifications of the machine.

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
        print("\nERROR: ECLAIR: ECLAIR_core: get_chunk_size: "
              "this machine does not have enough free memory resources "
              "to perform ensemble clustering.\n")
        exit(1)


def KMEANS(data, k):

    from sklearn.cluster import k_means, MiniBatchKMeans

    if data.shape[0] < 50000:
        centroids, cluster_labels, _ = k_means(data, k, init = 'k-means++', precompute_distances = 'auto', n_init = 20, max_iter = 300, n_jobs = 1)
    else:
        mbkm = MiniBatchKMeans(k, 'k-means++', max_iter = 300, batch_size = data.shape[0] / k, n_init = 20)
        mbkm.fit(data)
            
        centroids = mbkm.cluster_centers_
        cluster_labels = mbkm.labels_

    return centroids, cluster_labels


def hierarchical_clustering(data, n_clusters):

    from .Scalable_SLINK import SLINK
    from scipy.cluster.hierarchy import fcluster

    assert isinstance(n_clusters, int) and n_clusters > 1

    linkage_matrix = SLINK(data) 
    cluster_labels = fcluster(linkage_matrix, n_clusters - 1, 'maxclust')

    return cluster_labels


def toHDF5(hdf5_file_name, data_file_name, expected_rows, sampling_fraction,
           clustering_parameters, skip_rows, cell_IDs_column,
           extra_excluded_columns, time_info_column, scaling = False, 
           PCA_flag = False, N_PCA = 10):
    """Read the data and store it in HDF5 format. 
       Also records the cell/sample names. 
       If applicable, create spaces in this data structure 
       for various arrays involved 
       in affinity propagation clustering.

    Parameters
    ----------
    hdf5_file_name : string or file object

    data_file_name : string or file object

    expected_rows : int

    sampling_fraction : float

    clustering_parameters : namedtuple

    skip_rows : int

    cell_IDs_column : int

    extra_excluded_column : list

    scaling : Boolean, optional (default = True)
    
    Returns
    -------
    data : array (n_samples, n_features) 

    cell_IDs : array (n_samples,)

    hdf5_file_name : file object or string
    """
  
    assert isinstance(expected_rows,int)
    assert isinstance(sampling_fraction, float) or isinstance(sampling_fraction, int)
    assert isinstance(skip_rows, int)
    assert isinstance(cell_IDs_column, int)
    
    cell_IDs, time_info, data = dataProcessor(data_file_name, skip_rows,
               cell_IDs_column, extra_excluded_columns, time_info_column)

    unexpressed_indices = reportUnexpressed(data)
    if unexpressed_indices.size != 0:
        data = np.delete(data, unexpressed_indices, axis = 1)
        cell_IDs = np.delete(cell_IDs, unexpressed_indices)
        if time_info_column > 0:
            time_info = np.delete(time_info, unexpressed_indices)
        # Done with detecting unexpressed genes or reporters.

    method = clustering_parameters.clustering_method

    if scaling or method == 'DBSCAN':
        data = StandardScaler().fit_transform(data)

    if PCA_flag:
        if not scaling:
            data = StandardScaler().fit_transform(data)
        pca = PCA(copy = True)
        data = pca.fit_transform(data)[:, :N_PCA]

    N_samples = data.shape[0]

    subsample_size = int(floor(N_samples * sampling_fraction))

    # Create an HDF5 data structure. For data-sets with a large number of samples, ensemble clustering needs to handle certain arrays that possibly do not fit into memory. We store them on disk, along with the set of probability distributions of distances that we compute for each pair of clusters from the final consensus clustering. Similarly, this HDF5 file format is used for our implementation of affinity propagation clustering.
    with tables.open_file(hdf5_file_name, mode = 'w') as fileh:
        consensus_group = fileh.create_group(fileh.root, "consensus_group")

        atom = tables.Float32Atom()

        if method == 'affinity_propagation':
            aff_prop_group = fileh.create_group(fileh.root, "aff_prop_group")

            fileh.create_carray(aff_prop_group, 'availabilities', atom, (subsample_size, subsample_size), 'Matrix of availabilities for affinity propagation', filters = None)

            fileh.create_carray(aff_prop_group, 'responsibilities', atom, (subsample_size, subsample_size), 'Matrix of responsibilities for affinity propagation', filters = None)

            fileh.create_carray(aff_prop_group, 'similarities', atom, (subsample_size, subsample_size), 'Matrix of similarities for affinity propagation', filters = None)

            fileh.create_carray(aff_prop_group, 'temporaries', atom, (subsample_size, subsample_size), 'Matrix of temporaries for affinity propagation', filters = None)

            fileh.create_carray(aff_prop_group, 'parallel_updates', atom, (subsample_size, clustering_parameters.convergence_iter), 'Matrix of parallel updates for affinity propagation', filters = None)   

    return data, cell_IDs, time_info, hdf5_file_name


def dataProcessor(data_file_name, skip_rows, cell_IDs_column, 
                  extra_excluded_columns = None, time_info_column = -1):
    """Read the contents of data_file_name, extracting the names 
       or IDs of each sample, as stored in column labelled
       'cell_IDs_column' and creating an array of the features, 
       excluding those stored in 'extra_excluded_columns'. 
       Also check the validity of the excluded indices.

    Parameters
    ----------
    data_file_name : file object or string
   
    skip_rows : int

    cell_IDs_column : int

    extra_excluded_columns : list, optional (default = None)

    Returns
    -------
    cell_IDs : array (n_samples,)

    data : array (n_samples, n_features)
    """

    assert isinstance(skip_rows, int)
    assert isinstance(cell_IDs_column, int)
    assert isinstance(time_info_column, int)

    assert time_info_column != cell_IDs_column

    try:
        isinstance(skip_rows, int) and skip_rows >= 0
    except TypeError:
        time_now = datetime.datetime.today()
        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))
        print('\nECLAIR\t ERROR\t {}: the number of rows to skip as part of a header must be a non-negative integer.\n'.format(time_now))
        raise

    try:
        isinstance(cell_IDs_column, int) and cell_IDs_column >= 0
    except TypeError:
        time_now = datetime.datetime.today()
        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))
        print('\nECLAIR\t ERROR\t {}: the label distinguishing the column of cell IDs from other features must be a single integer.\n'.format(time_now))
        raise

    cell_IDs_column = [cell_IDs_column]    

    with open(data_file_name, 'r') as f:
        lst = f.readline()
        lst = lst.replace('\t', ' ').replace(',', ' ').split()
        N_cols = len(lst)

    assert time_info_column < N_cols

    with open(data_file_name, 'r') as f:
        cell_IDs = np.loadtxt(f, dtype = str, delimiter = '\t', 
                      skiprows = skip_rows, usecols = cell_IDs_column)

    if time_info_column > 0:
        time_info_column = [time_info_column]

        with open(data_file_name, 'r') as f:
            time_info = np.loadtxt(f, dtype = float, delimiter = '\t',
                          skiprows = skip_rows, usecols = [time_info_column])
    else:
        time_info_column = []
        time_info = np.zeros(0, dtype = float)

    if extra_excluded_columns is None:
        extra_excluded_columns = np.empty(0, dtype = int)
    else:
        extra_excluded_columns = np.array(extra_excluded_columns, dtype = int, 
                                          copy = False)
        extra_excluded_columns = np.clip(np.append(extra_excluded_columns, cell_IDs_column), 0, N_cols - 1)

        extra_excluded_columns = np.unique(extra_excluded_columns)

        ID_index = np.where(extra_excluded_columns == cell_IDs_column[0])[0]
        if ID_index.size != 0:
            extra_excluded_columns = np.delete(extra_excluded_columns, ID_index)

        if len(time_info_column) > 0:
            time_index = np.where(extra_excluded_columns == time_info_column[0])[0]
            if time_index.size != 0:
                extra_excluded_columns = np.delete(extra_excluded_columns, time_index)

    indices = np.delete(np.arange(N_cols), np.concatenate((extra_excluded_columns, cell_IDs_column, time_info_column)).astype(int))
    my_iterator = iter(indices)

    with open(data_file_name, 'r') as f:
        data = np.loadtxt(f, dtype = float, delimiter = '\t', 
                          skiprows = skip_rows, usecols = my_iterator)
        
    return cell_IDs, time_info, data 


def reportUnexpressed(data):
    """If a gene is unexpressed throughout all samples, 
       remove its index from the data-set.

    Parameters
    ----------
    data : array (n_samples, n_features)    
    """

    return np.where(data.sum(axis = 0) == 0)[0]


def build_weighted_adjacency(graph):

    adjacency_matrix = np.asarray(graph.get_adjacency(type = igraph.GET_ADJACENCY_BOTH).data)
    adjacency_matrix = adjacency_matrix.astype(dtype = float)

    N_clusters = adjacency_matrix.shape[0]
    c = 0
    for i in xrange(N_clusters - 1):
        for j in xrange(i + 1, N_clusters):
            if adjacency_matrix[i, j]:
                x = graph.es['weight'][c]
                adjacency_matrix[i, j] = x
                adjacency_matrix[j, i] = x
                c += 1

    return adjacency_matrix


def handle_off_diagonal_zeros(M):

    eensy = np.finfo(np.float32).eps * 1000

    M = np.array(M, copy = False)
    n = M.shape[0]

    zeros = np.where(M == 0)
    for i in xrange(zeros[0].size):
        if zeros[0][i] != zeros[1][i]:
            M[zeros[0][i], zeros[1][i]] = eensy
    M[np.diag_indices(n)] = 0


def get_MST(run, exemplars, exemplars_similarities, cell_IDs, name_tag,
            output_directory):
    """Build the minimum spanning tree of the graph whose nodes 
       are given by 'exemplars' and 'cell_IDs' and whose edges 
       are weighted according to the second argument, 
       a matrix of similarities. 
       Plot the MST and returns its structure.

    Parameters
    ----------
    run : int

    exemplars : array (n_clusters,)

    exemplars_similarities : array (n_clusters, n_clusters)

    cell_IDs : list

    name_tag : string

    Returns
    -------
    A sparse adjacency matrix for the spanning tree in CSR format
    """

    assert isinstance(run, int)
    assert isinstance(name_tag, str)

    n = len(exemplars)

    handle_off_diagonal_zeros(exemplars_similarities)

    g = igraph.Graph.Weighted_Adjacency(exemplars_similarities.tolist(),
                                  mode=igraph.ADJ_UPPER, attr = 'weight')

    g.vs['label'] = [cell_IDs[exemplar] for exemplar in exemplars]

    mst = g.spanning_tree(weights = g.es['weight'])

    layout = mst.layout('fr')
    name = output_directory + '/ECLAIR_figures/{}/mst-run-{}__{}.pdf'.format(name_tag, run + 1, name_tag)

    igraph.plot(mst, name, bbox = (5500, 5500), margin = 60, layout = layout,
                vertex_label_dist = 1)
    
    mst_adjacency_matrix = np.asarray(mst.get_adjacency(type = igraph.GET_ADJACENCY_BOTH).data)

    return scipy.sparse.csr_matrix(mst_adjacency_matrix)


def get_median(values, counts):

    N_pairs = np.sum(counts)
    median_index = (N_pairs + 1) / 2 if N_pairs % 2 else N_pairs / 2

    cumsum = 0
    for i, v in enumerate(values):
        cumsum += counts[i]
        if cumsum >= median_index:
            if N_pairs % 2:
                median = v
                break
            if N_pairs % 2 == 0 and cumsum >= median_index + 1:
                median = v
                break
            else:
                median = int(np.rint((v + values[i + 1]) / 2))
                break

    return median


def tree_path_integrals(hdf5_file_name, N_runs, cluster_dims_list, consensus_labels,
                        mst_adjacency_list, markov = False):
    """For each pair of cluster from the final ensemble clustering, 
       compute a distribution of distances as follows. 
       Assume that n_A cells are grouped into cluster A and n_B 
       into cluster B. Given cell 'a' from group A, 
       for each cell 'b' from group B, collect the distances 
       separating the cluster where 'a' resides in run 'i' 
       from the cluster where 'b' belongs in run 'i'; 
       do so for each of the 'N_runs' separate runs of subsampling 
       and clusterings. 

    Parameters
    ----------
    hdf5_file_name : string or file object

    N_runs : int

    cluster_dims_list : list of size N_clusters

    consensus_labels : list or array (n_samples)

    mst_adjacency_list : list of arrays 

    Returns
    -------
    consensus_means : array (N_clusters, N_clusters)

    consensus_variances : array (N_clusters, N_clusters)
    """

    assert isinstance(N_runs, int) and N_runs > 0

    hypergraph_adjacency = CE.load_hypergraph_adjacency(hdf5_file_name)
    cluster_runs_adjacency = hypergraph_adjacency.transpose().tocsr()

    del hypergraph_adjacency

    consensus_labels = np.asarray(consensus_labels)

    N_samples = consensus_labels.size

    N_clusters = np.unique(consensus_labels).size

    consensus_means = np.zeros((N_clusters, N_clusters), dtype = float)
    consensus_variances = np.zeros((N_clusters, N_clusters), dtype = float)
    consensus_medians = np.zeros((N_clusters, N_clusters), dtype = int)

    fileh = tables.open_file(hdf5_file_name, 'r+')

    consensus_distributions_values = fileh.create_vlarray(fileh.root.consensus_group, 'consensus_distributions_values', tables.UInt16Atom(), 'For each clusters a, b > a from ensemble clustering, stores the possible time steps it takes to reach cells from cluster a to cells from cluster b, over the possible causal sets associated to an ensemble of partitions', filters = None, expectedrows = N_clusters * (N_clusters - 1) / 2)
    
    consensus_distributions_counts = fileh.create_vlarray(fileh.root.consensus_group, 'consensus_distributions_counts', tables.Float128Atom(), 'For each clusters a, b > a from ensemble clustering, stores for each time step the number of its occurrences, weighted by the probability that such a time step takes place on the trees from the ensemble', filters = None, expectedrows = N_clusters * (N_clusters - 1) / 2)    

    cluster_separators = np.cumsum(cluster_dims_list)

    for a in xrange(N_clusters - 1):
        cells_in_a = np.where(consensus_labels == a)[0]

        for b in xrange(a+1, N_clusters):
            cells_in_b = np.where(consensus_labels == b)[0]

            count = 0

            counts_dict = defaultdict(int)
            weights_dict = defaultdict(float)
 
            for run in xrange(N_runs):
                if cells_in_a.size == 1 and (cluster_separators[run + 1] - cluster_separators[run]) == 1:
                    single_elt = cluster_runs_adjacency[cells_in_a, cluster_separators[run]]
                    if single_elt == 1:
                        cluster_IDs_a = np.zeros(1, dtype = np.int32)
                    else:
                        cluster_IDs_a = np.zeros(0, dtype = np.int32)
                else:
                    try:
                        cluster_IDs_a = np.where(np.squeeze(np.asarray(cluster_runs_adjacency[cells_in_a, cluster_separators[run]:cluster_separators[run + 1]].todense())) == 1)
                    except ValueError:
                        continue

                    if isinstance(cluster_IDs_a, tuple):
                        if len(cluster_IDs_a) == 1:
                            if (cluster_separators[run + 1] - cluster_separators[run]) == 1:
                                cluster_IDs_a = np.zeros(cluster_IDs_a[0].size, dtype = np.int32)
                            else:
                                cluster_IDs_a = cluster_IDs_a[0]
                        else:
                            cluster_IDs_a = cluster_IDs_a[1]
                
                if cluster_IDs_a.size == 0:
                    continue

                if cells_in_b.size == 1 and (cluster_separators[run + 1] - cluster_separators[run]) == 1:
                    single_elt = cluster_runs_adjacency[cells_in_b, cluster_separators[run]]
                    if single_elt == 1:
                        cluster_IDs_b = np.zeros(1, dtype = np.int32)
                    else:
                        cluster_IDs_b = np.zeros(0, dtype = np.int32)
                else:
                    try:
                        cluster_IDs_b = np.where(np.squeeze(np.asarray(cluster_runs_adjacency[cells_in_b, cluster_separators[run]:cluster_separators[run + 1]].todense())) == 1)
                    except ValueError:
                        continue

                    if isinstance(cluster_IDs_b, tuple):
                        if len(cluster_IDs_b) == 1:
                            if (cluster_separators[run + 1] - cluster_separators[run]) == 1:
                                cluster_IDs_b = np.zeros(cluster_IDs_b[0].size, dtype = np.int32)        
                            else:
                                cluster_IDs_b = cluster_IDs_b[0]            
                        else:
                            cluster_IDs_b = cluster_IDs_b[1]

                if cluster_IDs_b.size == 0:
                    continue
                
                cluster_IDs_a, counts_a = np.unique(cluster_IDs_a, 
                                              return_counts = True)    

                cluster_IDs_b, counts_b = np.unique(cluster_IDs_b, 
                                              return_counts = True)

                n_a = cluster_IDs_a.size
                n_b = cluster_IDs_b.size

                mst_time_steps_matrix = scipy.sparse.csgraph.dijkstra(mst_adjacency_list[run], directed = False, unweighted = True)

                #x = mst_time_steps_matrix[cluster_IDs_a]
                #y = np.zeros((mst_time_steps_matrix.shape[1], n_b), dtype = np.int32)
                #y[(cluster_IDs_b, xrange(n_b))] = 1
                #time_steps_values = np.dot(x, y)
                
                time_steps_values = mst_time_steps_matrix[cluster_IDs_a]
                time_steps_values = time_steps_values[:, cluster_IDs_b]
                time_steps_values = time_steps_values.astype(int)

                time_steps_counts = np.dot(counts_a.reshape(-1, 1), counts_b.reshape(1, -1))
                
                if markov:
                    mst_adjacency = np.squeeze(np.asarray(mst_adjacency_list[run].todense()))
                    Delta = np.sum(mst_adjacency, axis = 1).reshape(-1, 1).astype(float)
                    transition_probabilities = np.divide(mst_adjacency, Delta)
                
                    time_steps_probabilities = np.zeros((n_a, n_b), 
                                                     dtype = float)
                    for time_step in np.unique(time_steps_values):
                        idx = np.where(time_steps_values == time_step)
                    
                        new_x = map(lambda i: cluster_IDs_a[i], idx[0])
                        new_x = np.array(new_x, dtype = int)
                    
                        new_y = map(lambda i: cluster_IDs_b[i], idx[1])
                        new_y = np.array(new_y, dtype = int)
                    
                        mapped_idx = (new_x, new_y)
                        time_reversed_mapped_idx = (new_y, new_x)
                    
                        if time_step == 0:
                            time_steps_probabilities[idx] = 1
                        elif time_step == 1:
                            tmp_1 = transition_probabilities[mapped_idx]
                            tmp_2 = transition_probabilities[time_reversed_mapped_idx]
                        
                            time_steps_probabilities[idx] = np.minimum(tmp_1, tmp_2)
                        else:
                            markov_chain = np.linalg.matrix_power(transition_probabilities, time_step)
                            tmp_1 = markov_chain[mapped_idx]
                            tmp_2 = markov_chain[time_reversed_mapped_idx]
                        
                            time_steps_probabilities[idx] = np.minimum(tmp_1, tmp_2)
                    
                    time_steps_weights = time_steps_counts * time_steps_probabilities
                    
                else:
                    time_steps_weights = time_steps_counts

                time_steps_counts = np.ravel(time_steps_counts)
                time_steps_values = np.ravel(time_steps_values)
                time_steps_weights = np.ravel(time_steps_weights)
                for i in xrange(n_a * n_b):
                    counts_dict[time_steps_values[i]] += time_steps_counts[i]
                    weights_dict[time_steps_values[i]] += time_steps_weights[i]
                        
                count += 1 

            if count > 0:
                time_steps = np.array(counts_dict.keys(), dtype = int)
                counts = np.array(counts_dict.values(), dtype = int)
                weights = np.array(weights_dict.values(), dtype = float)
                
                consensus_distributions_values.append(time_steps)
                consensus_distributions_counts.append(weights)
                
                median = get_median(time_steps, counts)
                consensus_medians[a, b] = median
                consensus_medians[b, a] = median
            
                normalization_factor = weights.sum()
                weights /= float(normalization_factor)
                
                mean = np.inner(time_steps, weights)
                consensus_means[a, b] = mean
                consensus_means[b, a] = mean

                variance = np.inner((time_steps - mean) ** 2, weights)
                consensus_variances[a, b] = variance 
                consensus_variances[b, a] = variance
            else:
                consensus_distributions_values.append([0])
                consensus_distributions_counts.append([0])
                
                consensus_medians[a, b] = np.nan
                consensus_medians[b, a] = np.nan
            
                consensus_means[a, b] = np.nan
                consensus_means[b, a] = np.nan

                consensus_variances[a, b] = np.nan
                consensus_variances[b, a] = np.nan

    fileh.close()

    return consensus_medians, consensus_means, consensus_variances


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


def handle_disconnected_graph(consensus_medians, consensus_means,
                              consensus_variances, consensus_labels):

    n = consensus_means.shape[0]
    assert n == np.amax(consensus_labels) + 1

    means_nan_ID = np.empty((n, n), dtype = int)
    np.isnan(consensus_means, means_nan_ID)

    null_means = np.where(means_nan_ID.sum(axis = 1) == n - 1)

    variances_nan_ID = np.empty((n, n), dtype = int)
    np.isnan(consensus_variances, variances_nan_ID)

    null_variances = np.where(variances_nan_ID.sum(axis = 1) == n - 1)

    medians_nan_ID = np.empty((n, n), dtype = int)
    np.isnan(consensus_medians, medians_nan_ID)

    null_medians = np.where(medians_nan_ID.sum(axis = 1) == n - 1)

    if not (np.allclose(null_means[0], null_variances[0]) and np.allclose(null_means[0], null_medians[0])):
        time_now = datetime.datetime.today()

        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))
        raise ValueError('\nECLAIR\t ERROR\t {}: serious problem with the noise clusters: mismatch between the computations of means and variances.\n'.format(time_now))

    exclude_nodes = null_means[0]

    consensus_medians = np.delete(consensus_medians, exclude_nodes, axis = 0)
    consensus_medians = np.delete(consensus_medians, exclude_nodes, axis = 1)

    consensus_means = np.delete(consensus_means, exclude_nodes, axis = 0)
    consensus_means = np.delete(consensus_means, exclude_nodes, axis = 1)

    consensus_variances = np.delete(consensus_variances, exclude_nodes, axis = 0)
    consensus_variances = np.delete(consensus_variances, exclude_nodes, axis = 1)
   
    for elt in exclude_nodes:
        consensus_labels[np.where(consensus_labels == elt)[0]] = n

    consensus_labels = one_to_max(consensus_labels)
    consensus_labels[np.where(consensus_labels == n - exclude_nodes.size)[0]] = -1

    cells_kept = np.where(consensus_labels != -1)[0]  

    return consensus_medians, consensus_means, consensus_variances, consensus_labels, cells_kept, exclude_nodes
      

def MST_coloring(data, consensus_labels):

    data = StandardScaler().fit_transform(data)
    pca = PCA(copy = True)
    rotated_data = pca.fit_transform(data)

    N_clusters = np.amax(consensus_labels) + 1

    cc_centroids = np.zeros((N_clusters, data.shape[1]), dtype = float)
    for cluster_ID in xrange(N_clusters):
        cc_centroids[cluster_ID] = np.median(rotated_data[np.where(consensus_labels == cluster_ID)[0]], axis = 0) 

    cc_centroids = pca.transform(cc_centroids)

    min_PCA = np.amin(cc_centroids[:, 0:3])
    max_PCA = np.amax(cc_centroids[:, 0:3])
    cc_RGB = np.divide(cc_centroids[:, 0:3] - min_PCA, max_PCA - min_PCA)
    cc_RGB *= 255
    cc_RGB = np.rint(cc_RGB)
    cc_RGB = cc_RGB.astype(dtype = int)
    cc_RGB = np.clip(cc_RGB, 0, 255)

    cc_colors = ["rgb({}, {}, {})".format(cc_RGB[i, 0], cc_RGB[i, 1], cc_RGB[i, 2]) for i in xrange(N_clusters)]

    return cc_colors


def DBSCAN_LOAD(hdf5_file_name, data, subsamples_matrix, clustering_parameters):

    assert clustering_parameters.clustering_method == 'DBSCAN'

    minPts = clustering_parameters.minPts
    eps = clustering_parameters.eps
    quantile = clustering_parameters.quantile
    metric = clustering_parameters.metric

    return DBSCAN_multiplex.load(hdf5_file_name, data, minPts, eps, quantile,
                                 subsamples_matrix, metric = metric)


def subsamples_clustering(hdf5_file_name, data, sampled_indices, clustering_parameters, run):

    time_now = datetime.datetime.today()
    format = "%Y-%m-%d %H:%M:%S"
    time_now = str(time_now.strftime(format))
    print('ECLAIR\t INFO\t {}: starting run of clustering number {:n}.'.format(time_now, run + 1))

    beg_clustering = time.time()

    method = clustering_parameters.clustering_method

    if method == 'affinity_propagation':
        # Perform affinity propagation clustering with our customized module,
        # computing and storing to disk the similarities matrix 
        # between those sampled cells, as a preliminary step.
        args = "Concurrent_AP -f {0} -c {1} -i {2}".format(hdf5_file_name,
            clustering_parameters.convergence_iter, clustering_parameters.max_iter)
        subprocess.call(args, shell = True)
        
        with open('./concurrent_AP_output/cluster_centers_indices.tsv', 'r') as f:
            cluster_centers_indices = np.loadtxt(f, dtype = float, delimiter = '\t')
            
        with open('./concurrent_AP_output/labels.tsv', 'r') as f:
            cluster_labels = np.loadtxt(f, dtype = float, delimiter = '\t')

        exemplars = [sampled_indices[i] for i in cluster_centers_indices] 
        # Exemplars: indices as per the full data-set 
        # that are the best representatives of their respective clusters
        centroids = data[exemplars, :]

    elif method == 'DBSCAN':
        sampled_data = np.take(data, sampled_indices, axis = 0)

        minPts = clustering_parameters.minPts 
        metric = clustering_parameters.metric       

        _, cluster_labels = DBSCAN_multiplex.shoot(hdf5_file_name, minPts, run,
                                            random_state = None, verbose = True) 

        cluster_IDs = np.unique(cluster_labels)
        cluster_IDs = np.extract(cluster_IDs != -2, cluster_IDs)
        cluster_IDs = np.extract(cluster_IDs != -1, cluster_IDs)

        centroids = np.empty((0, data.shape[1]), dtype = float)
        for label in cluster_IDs:
            indices = np.where(cluster_labels == label)[0]
            centroids = np.append(centroids, 
                np.reshape(data[indices].mean(axis = 0), newshape = (1, -1)), 
                    axis = 0)

        closest_to_centroids, _ = pairwise_distances_argmin_min(centroids,
                                             sampled_data, metric = metric)

        exemplars = [sampled_indices[i] for i in closest_to_centroids] 
        # Exemplars: indices as per the full data-set that are the best
        # representatives of their respective clusters

    elif method == 'hierarchical':
        sampled_data = np.take(data, sampled_indices, axis = 0)

        n_clusters = clustering_parameters.k

        cluster_labels = hierarchical_clustering(sampled_data, n_clusters)

        centroids = np.empty((0, data.shape[1]), dtype = float)

        cluster_IDs = np.unique(cluster_labels)
        for label in cluster_IDs:
            indices = np.where(cluster_labels == label)[0]
            centroids = np.append(centroids, 
                np.reshape(np.median(data[indices], axis = 0), newshape = (1, -1)),
                    axis = 0)

        closest_to_centroids, _ = pairwise_distances_argmin_min(centroids,
                                        sampled_data, metric = 'manhattan')

        exemplars = [sampled_indices[i] for i in closest_to_centroids]

    elif method == 'k-means':
        sampled_data = np.take(data, sampled_indices, axis = 0)

        n_clusters = clustering_parameters.k

        centroids, cluster_labels = KMEANS(sampled_data, n_clusters)

        closest_to_centroids, _ = pairwise_distances_argmin_min(centroids, sampled_data, batch_size = min(centroids.shape[0], get_chunk_size(sampled_data.shape[1], 4)))
 
        exemplars = [sampled_indices[i] for i in closest_to_centroids] 
        # Exemplars: indices as per the full data-set that are the best
        # representatives of their respective clusters

    end_clustering = time.time()

    time_now = datetime.datetime.today()
    format = "%Y-%m-%d %H:%M:%S"
    time_now = str(time_now.strftime(format))
    print('ECLAIR\t INFO\t {}: done with this round of clustering; it took {:n} seconds.'.format(time_now, round(end_clustering - beg_clustering, 4)))

    return exemplars, centroids, cluster_labels


def get_valid_rows(cluster_runs, N_samples):

    valid_rows = np.where(np.sum(np.isnan(cluster_runs).astype(dtype = int), axis = 1) != N_samples)[0]

    if len(valid_rows) == 0:
        time_now = datetime.datetime.today()
        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))

        print('\n')
        raise ValueError("\nECLAIR\t WARNING\t {}: all the clusters generated consist of noise. Please start over with a different choice of parameter(s).\n".format(time_now))
        return 'ERROR'

    return cluster_runs[valid_rows]


def fill_cluster_runs_matrix(holder, sampled_indices, cluster_labels, centroids,
                             exemplars, cell_IDs, data, mst_adjacency_list,
                             output_directory, cluster_runs, upsampling = True):

    N_samples = holder.N_samples
    subsample_size = holder.subsample_size
    N_runs = holder.N_runs
    name_tag = holder.name_tag
    method = holder.method
    run = holder.run
    error_count = holder.error_count

    if method in {'affinity_propagation', 'hierarchical', 'k-means'}:
        cluster_runs[run, sampled_indices] = cluster_labels

    if method == 'DBSCAN':
        cluster_runs[run] = cluster_labels

        noise_indices = np.where(cluster_runs[run] == -1)[0]
        cluster_runs[run, noise_indices] = np.nan
      
        not_selected = np.where(cluster_runs[run] == -2)[0]
        cluster_runs[run, not_selected] = np.nan

    metric = 'manhattan' if (method == 'hierarchical') else 'euclidean'

    if upsampling:
        left_overs = np.isnan(cluster_runs[run])

        M = data[left_overs]
        closest_index, _ = pairwise_distances_argmin_min(M, centroids, metric = metric, batch_size = min(centroids.shape[0], get_chunk_size(M.shape[1], 4)))

        cluster_runs[run, left_overs] = closest_index

    if np.sum(np.isnan(cluster_runs[run]).astype(dtype = int)) == N_samples:
        print("\nWARNING\t@ ECLAIR: this round of subsampling and clustering yielded a single cluster of cells labelled as noise. It will be discarded an replaced by another run of clustering on a new random subsample of the whole data-set.\n")

        error_count += 1
        holder = holder._replace(error_count = error_count)

        if error_count >= N_runs - 1:
            time_now = datetime.datetime.today()
            format = "%Y-%m-%d %H:%M:%S"
            time_now = str(time_now.strftime(format))

            print('\n')
            raise ValueError("\nECLAIR\t WARNING\t {}: too many noisy clusters have been generated. Please start over with a different choice of parameter(s).\n".format(time_now))
            return 'ERROR'

        if method in {'affinity_propagation', 'hierarchical', 'k-means'}:
            subsamples_matrix[run] = random.sample(xrange(N_samples), subsample_size)
        else:
            run += 1
            holder = holder._replace(run = run)

    else:
        if len(centroids > 0):
            exemplars_similarities = pairwise_distances(centroids, centroids,
                                                        metric, n_jobs = 1)

        if (len(exemplars) > 0) and (not np.all(exemplars_similarities == 0.0)):
            mst_adjacency_matrix = get_MST(run, exemplars, exemplars_similarities,
                                           cell_IDs, name_tag, output_directory)

            mst_adjacency_list.append(mst_adjacency_matrix)

            run += 1
            holder = holder._replace(run = run)
        else:
            time_now = datetime.datetime.today()
            format = "%Y-%m-%d %H:%M:%S"
            time_now = str(time_now.strftime(format))
            print("ECLAIR\t INFO\t {}: cannot build a spanning tree for this run of clustering. It will be replaced by another run of clustering on new random subsample of the data-set.".format(time_now))       

            error_count += 1
            holder = holder._replace(error_count = error_count)

            if error_count > N_runs - 1:
                time_now = datetime.datetime.today()
                format = "%Y-%m-%d %H:%M:%S"
                time_now = str(time_now.strftime(format))

                print('\n')
                raise ValueError("\nECLAIR\t WARNING\t {}: too many noisy clusters have been generated. Please start over with a different choice of parameter(s).\n".format(time_now))
                return 'ERROR'

            if method in {'affinity_propagation', 'k-means'}:
                subsamples_matrix[run] = random.sample(xrange(N_samples), subsample_size)
            else:
                run += 1
                holder = holder._replace(run = run)

    return holder


def output_ensemble_distances_distributions(hdf5_file_name, exclude_nodes,
                                            output_directory, name_tag):

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/ensemble_distances_distributions.txt'.format(name_tag), 'w') as f:
        fileh = tables.open_file(hdf5_file_name, 'r+')

        all_values = fileh.root.consensus_group.consensus_distributions_values
        all_counts = fileh.root.consensus_group.consensus_distributions_counts

        f.write('(cluster a, cluster b, a < b)\t\t\t(value of the distance, number of occurrences)\n')

        N_rows = all_values.nrows
        N_clusters = int((1 + sqrt(1 + 8 * N_rows) ) / 2) 
        # the number of nodes initially present, 
        # before any call to the procedure 'handle_disconnected_graph'

        c = 0
        for a in xrange(N_clusters - 1):
            if a in exclude_nodes:
                c += N_clusters - a - 1
                continue

            for b in xrange(a + 1, N_clusters):
                if b in exclude_nodes:
                    c += 1
                    continue
            
                values = all_values[c]
                counts = all_counts[c]

                n = values.size
                for i in xrange(n):
                    f.write('({}, {})\t'.format(values[i], counts[i]))
                f.write('\n')

                c += 1

        fileh.close()


def ECLAIR_processing(hdf5_file_name, data_info, clustering_parameters,
                      cc_parameters, output_directory, upsampling = True):
    """
    Parameters
    ----------
    hdf5_file_name : file object or string

    data_info : instance of 'Data_info' namedtuple

    clustering_parameters : namedtuple

    cc_parameters : namedtuple

    output_directory : file object or string

    Returns
    -------
    name_tag : string
    """

    N_runs = cc_parameters.N_runs
    sampling_fraction = cc_parameters.sampling_fraction

    assert isinstance(data_info.expected_N_samples, int)
    assert isinstance(N_runs, int)
    assert isinstance(sampling_fraction, float) or isinstance(sampling_fraction, int)
    assert isinstance(data_info.skip_rows, int)

    beg = time.time()

    # Converting the data-set to HDF5 format:
    data, cell_IDs, time_info, hdf5_file_name = toHDF5(hdf5_file_name,
        data_info.data_file_name, data_info.expected_N_samples, sampling_fraction,
        clustering_parameters, data_info.skip_rows, data_info.cell_IDs_column,
        data_info.extra_excluded_columns, data_info.time_info_column)

    N_samples, N_features = data.shape
    subsample_size = int(floor(N_samples * sampling_fraction))

    # Creating a directory to hold the minimum-spanning trees corresponding to each of the 'N_runs' independent clusterings, along with the final lineage tree built via consensus clustering from this ensemble:
    try:
        os.makedirs(output_directory)
    except OSError:
        if not os.path.isdir(output_directory):
            time_now = datetime.datetime.today()
            format = "%Y-%m-%d %H:%M:%S"
            time_now = str(time_now.strftime(format))
            print('\nECLAIR\t ERROR\t {}\n'.format(time_now))
            raise

    try:
        os.makedirs(output_directory + '/ECLAIR_figures')

        time_now = datetime.datetime.today()
        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))
        print("ECLAIR\t INFO\t {}: images of the minimum-spanning trees associated with each of the {:n} of clusterings are stored in a newly-created folder 'figures' in your current working directory.".format(time_now, N_runs))
    except OSError:
        if not os.path.isdir(output_directory + '/ECLAIR_figures'):
            print('\nECLAIR\t ERROR\t {}\n'.format(time_now))
            raise

    right_now = datetime.datetime.today()
    format = "%Y-%m-%d__%H:%M:%S"
    name_tag = str(right_now.strftime(format))

    try:
        os.makedirs(output_directory + '/ECLAIR_figures/{}'.format(name_tag))
    except OSError:
        print("\nECLAIR\t ERROR\t {}: cannot create directory for storing the figures of the minimum-spanning trees associated to the current batch of {:n} independent clusterings!\n".format(name_tag, N_runs))
        raise

    method = clustering_parameters.clustering_method     
    metric = 'manhattan' if (method == 'hierarchical') else 'euclidean'
    kernel_mult = 2.0 if N_samples < 5000 else 5.0

    local_densities = Density_Sampling.get_local_densities(data, kernel_mult, metric)

    subsamples_matrix = np.zeros((N_runs, subsample_size), dtype = np.int32)
    for run in xrange(N_runs):
        samples_kept = Density_Sampling.density_sampling(data, local_densities,
                          metric, kernel_mult, desired_samples = subsample_size)
        
        N_kept = len(samples_kept)

        if N_kept > subsample_size:
            samples_kept.sort()
            discard_ind = random.sample(xrange(N_kept), N_kept - subsample_size)
            samples_kept = np.delete(samples_kept, discard_ind)

        elif N_kept < subsample_size:
            select_from = np.setdiff1d(np.arange(N_samples), samples_kept,
                                       assume_unique = True)
            add_ind = random.sample(select_from, subsample_size - N_kept)
            samples_kept = np.append(samples_kept, add_ind)
            samples_kept.sort()
        else:
            samples_kept.sort()

        subsamples_matrix[run] = samples_kept

    # The 'cluster_runs' array records the cluster IDs of each cell,
    # for each of 'N_runs' of independent clusterings. 
    # Entries associated to cells not part of a subsample at a given run are marked 'NaN'.
    cluster_runs = np.full((N_runs, N_samples), np.nan)
        
    mst_adjacency_list = []

    # START LOOP. Beginning of tree runs loop:
    beg_run = time.time()

    method = clustering_parameters.clustering_method

    if method == 'DBSCAN':
        global eps
        eps = DBSCAN_LOAD(hdf5_file_name, data, subsamples_matrix, clustering_parameters)

    holder = Holder(N_samples, subsample_size, N_runs, name_tag, method, run = 0,
                    error_count = 0)

    while holder.run < N_runs:
        sampled_indices = subsamples_matrix[run]

        exemplars, centroids, cluster_labels = subsamples_clustering(hdf5_file_name,
            data, sampled_indices, clustering_parameters, holder.run)  

        # Fill-in 'cluster_runs', a matrix mapping each run to the corresponding
        # cluster identitiers of each of the N_samples records. 
        # A cell not belonging to any cluster would be marked NaN; this does
        # apply if we are upsampling each of the cells to their nearest cluster. 
        holder = fill_cluster_runs_matrix(holder, sampled_indices, cluster_labels,
                                          centroids, exemplars, cell_IDs, data,
                                          mst_adjacency_list, output_directory,
                                          cluster_runs, upsampling)
        
    end_run = time.time()
    # END LOOP.

    if method == 'DBSCAN':
        cluster_runs = get_valid_rows(cluster_runs, N_samples)
        N_runs = cluster_runs.shape[0]

    beg_ens = time.time()

    try:
        os.makedirs(output_directory + '/ECLAIR_ensemble_clustering_files')

        time_now = datetime.datetime.today()
        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))
        print("ECLAIR\t INFO\t {}: files of the means, variances associated to each set of pairwise distances are stored in 'ensemble_clustering_files' in your current working directory. This folder also contains a file of each distribution of pairwise distances, a file of the ensemble cluster IDs for each sample and a file of the cluster IDs for the different runs of subsampling and clustering.".format(time_now))
    except OSError:
        if not os.path.isdir(output_directory + '/ECLAIR_ensemble_clustering_files'):
            print('\nECLAIR\t ERROR\t {}\n'.format(time_now))
            raise

    try:
        os.makedirs(output_directory + '/ECLAIR_ensemble_clustering_files/{}'.format(name_tag))
    except OSError:
        time_now = datetime.datetime.today()
        format = "%Y-%m-%d %H:%M:%S"
        time_now = str(time_now.strftime(format))
        print('\nECLAIR\t ERROR\t {}: cannot create a sub-directory for storing the afore-mentioned key information on ensemble clustering for this session!\n'.format(time_now))
        raise

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/cluster_runs.txt'.format(name_tag), 'w') as file:
        np.savetxt(file, cluster_runs, fmt = '%.1f', delimiter = ' ', 
                   newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/mst_adjacency_list.txt'.format(name_tag), 'w') as file:
        for run in xrange(len(mst_adjacency_list)):
            sparse_mat = mst_adjacency_list[run]

            file.write("RUN {}\n".format(run))
            np.savetxt(file, sparse_mat.indices, fmt = '%d', delimiter = ' ')
            file.write('\n')
            np.savetxt(file, sparse_mat.indptr, fmt = '%d', delimiter = ' ')
            file.write('\n')

    # Perform ensemble clustering and returns a vector of cluster IDs 
    # for each cell:
    consensus_labels = CE.cluster_ensembles(cluster_runs, hdf5_file_name, 
                           verbose = False, N_clusters_max = cc_parameters.N_cc)

    end_ens = time.time()

    # return the size of each of the clusters of ensemble clusterings; 
    # compute the mutual information between the consensus clustering 
    # and each separate run of clusterings; 
    # build the adjacency matrix for ensemble clustering.
    cluster_dims_list, mutual_info_list, _ = CE.overlap_matrix(hdf5_file_name,
                                                consensus_labels, cluster_runs)

    beg_dist = time.time()

    # compute the median, mean and the variance for each of the distributions
    # of distances separating each pair of clusters in ensemble clustering. 
    # The distribution is obtained via all underlying independent runs of
    # subsampling and clusterings.
    # A detailed explanation of how those distances are computed is provided 
    # in the documentation of our 'tree_path_integrals' procedure.
    consensus_medians, consensus_means, consensus_variances = tree_path_integrals(hdf5_file_name, N_runs, cluster_dims_list, consensus_labels, mst_adjacency_list) 

    end_dist = time.time() 

    time_now = datetime.datetime.today()
    format = "%Y-%m-%d %H:%M:%S"
    time_now = str(time_now.strftime(format))
    print("\nECLAIR\t INFO {}: done with computing the means and variances for the pair of distances along each minimum spanning tree between the groups of cells underlying each cluster of the ensemble clustering. The full probability distributions are available and stored in HDF5 format and available for further analysis.\n".format(time_now))

    consensus_medians, consensus_means, consensus_variances, consensus_labels, cells_kept, exclude_nodes = handle_disconnected_graph(consensus_medians,
                                consensus_means, consensus_variances,
                                consensus_labels)

    N_clusters = consensus_means.shape[0]
    assert N_clusters == np.amax(consensus_labels) + 1

    handle_off_diagonal_zeros(consensus_means)
    
    g = igraph.Graph.Weighted_Adjacency(consensus_means.tolist(),
                           mode=igraph.ADJ_UPPER, attr = 'weight')

    g.vs['label'] = xrange(N_clusters)
    # Building a weighted graph from the matrix of the mean 
    # of pairwise distances between clustered cells. 

    cc_colors = MST_coloring(data, consensus_labels)

    mst = g.spanning_tree(weights = g.es['weight'])
 
    layout = mst.layout('fr')
    name = output_directory + '/ECLAIR_figures/{}/consensus-lineage__{}.pdf'.format(name_tag, name_tag)  

    cluster_sizes = np.bincount(consensus_labels[cells_kept])

    vertex_sizes = N_clusters * 150 * np.divide(cluster_sizes.astype(float),
                                                np.sum(cluster_sizes))
    vertex_sizes += 50 + (N_clusters / 3)
    
    consensus_adjacency = np.asarray(mst.get_adjacency(type = igraph.GET_ADJACENCY_BOTH).data)
    consensus_adjacency = consensus_adjacency.astype(dtype = float)
    
    std = np.sqrt(consensus_variances)[np.triu_indices(N_clusters)]
    msk = (consensus_adjacency != 0)[np.triu_indices(N_clusters)]
    std = np.compress(msk, std)
    
    igraph.plot(mst, name, bbox = (200 * N_clusters, 200 * N_clusters), margin = 250,
                layout = layout, edge_width = (np.multiply(np.exp(-std), 
                6 * N_clusters) + (N_clusters / 3)).tolist(), vertex_label_dist = 0, 
                vertex_label_size = 50, vertex_size = vertex_sizes.tolist(),
                vertex_color = cc_colors)
    # Displaying the lineage inferred from the data-set.

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/consensus_labels.txt'.format(name_tag, name_tag), 'w') as file:
        np.savetxt(file, consensus_labels, fmt = '%d', delimiter = ' ', 
                   newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/consensus_adjacency_matrix.txt'.format(name_tag), 'w') as file:
        np.savetxt(file, consensus_adjacency, fmt = '%.1f', delimiter = ' ', 
                   newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/consensus_topological_distances_matrix.txt'.format(name_tag), 'w') as file:
        consensus_topological_distances_matrix = scipy.sparse.csgraph.dijkstra(consensus_adjacency, directed = False, unweighted = True)
        np.savetxt(file, consensus_topological_distances_matrix, fmt = '%.4f',
                   delimiter = ' ', newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/consensus_distances_matrix.txt'.format(name_tag), 'w') as file:
        weighted_consensus_adjacency = build_weighted_adjacency(mst)
        consensus_distances_matrix = scipy.sparse.csgraph.dijkstra(weighted_consensus_adjacency, directed = False, unweighted = False)
        np.savetxt(file, consensus_distances_matrix, fmt = '%.4f', 
                   delimiter = ' ', newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/ensemble_distances_medians.txt'.format(name_tag), 'w') as file:
        np.savetxt(file, consensus_medians, fmt = '%d', delimiter = ' ', 
                   newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/ensemble_distances_means.txt'.format(name_tag), 'w') as file:
        np.savetxt(file, consensus_means, fmt = '%.4f', delimiter = ' ', 
                   newline = '\n')

    with open(output_directory + '/ECLAIR_ensemble_clustering_files/{}/ensemble_distances_variances.txt'.format(name_tag), 'w') as file:
        np.savetxt(file, consensus_variances, fmt = '%.6f', delimiter = ' ', 
                   newline = '\n')

    output_ensemble_distances_distributions(hdf5_file_name, exclude_nodes,
                                            output_directory, name_tag)

    end = time.time()

    time_now = datetime.datetime.today()
    format = "%Y-%m-%d %H:%M:%S"
    time_now = str(time_now.strftime(format))
    print("ECLAIR\t INFO\t {}: the various runs of subsampling and clustering took {:n} seconds, ensemble clustering took {:n} seconds, while calculating the distribution of pairwise distances took {:n} seconds. The whole process lasted {:n} seconds".format(time_now, round(end_run - beg_run, 4), round(end_ens - beg_ens, 4), round(end_dist - beg_dist, 4), round(end - beg, 4)))


    print('\n*****************************************')
    print('*****************************************\n')

    return name_tag

