#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistical_performance/Scalable_SLINK.py

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


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


__all__ = ['SLINK']

  
def from_pointer_representation(Z, Lambda, Pi):
    """
    Build a linkage matrix from a pointer representation.
    
    Parameters
    ----------
    Z : array
        An array to store the linkage matrix
        
    Lambda : array
        The lambda array of the pointer representation
        
    Pi : array
        The Pi array associated to the pointer representation
    """
    
    N = Z.shape[0] + 1
    
    assert Lambda.size == N
    assert Pi.size == N
    
    sorted_indices = np.argsort(Lambda)
    node_IDs = np.arange(N)
    
    for i in xrange(N - 1):
        current_leaf = sorted_indices[i]
        
        pi = Pi[current_leaf]
        if node_IDs[current_leaf] < node_IDs[pi]:
            Z[i, 0] = node_IDs[current_leaf]
            Z[i, 1] = node_IDs[pi]
        else:
            Z[i, 0] = node_IDs[pi]
            Z[i, 1] = node_IDs[current_leaf]
            
        Z[i, 2] = Lambda[current_leaf]
        node_IDs[pi] = N + i
    
    Z[:, 3] = 0

    calculate_cluster_sizes(Z)


def calculate_cluster_sizes(Z):
    """
    Compute the size of each cluster. The result is the fourth column
    of the linkage matrix.
    
    Parameters
    ----------
    Z : array
        The linkage matrix. The fourth column can be empty.
    """

    N = Z.shape[0] + 1
    
    for i in xrange(N - 1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        
        if left_child >= N:
            Z[i, 3] += Z[left_child - N, 3]
        else:
            Z[i, 3] += 1
            
        if right_child >= N:
            Z[i, 3] += Z[right_child - N, 3]
        else:
            Z[i, 3] += 1


def SLINK(data):
    """
    The SLINK algorithm for single-linkage agglomerative clustering
    in quadratic time complexity.
    
    Parameters
    ----------
    data : array
        A dataset. To each row corresponds an observation.
        
    Returns
    -------
    Z : array
        A (n - 1) * 4 linkage matrix to store the results of SLINK computations.
    """

    N = data.shape[0]
    
    M = np.empty(N, dtype = float)
    Lambda = np.empty(N, dtype = float)
    Pi = np.empty(N, dtype = int)
    
    Pi[0] = 0
    Lambda[0] = np.inf
    
    for i in xrange(1, N):
        Pi[i] = i
        Lambda[i] = np.inf
        
        M[:i] = pairwise_distances(data[i], data[:i], metric = 'manhattan', n_jobs = -1).reshape(i)
            
        for j in xrange(i):
            if Lambda[j] >= M[j]:
                M[Pi[j]] = min(M[Pi[j]], Lambda[j])
                Lambda[j] = M[j]
                Pi[j] = i
            else:
                M[Pi[j]] = min(M[Pi[j]], M[j])
                
        for j in xrange(i):
            if Lambda[j] >= Lambda[Pi[j]]:
                Pi[j] = i
    
    Z = np.zeros((N - 1, 4), dtype = float)
    
    from_pointer_representation(Z, Lambda, Pi)
    
    return Z

