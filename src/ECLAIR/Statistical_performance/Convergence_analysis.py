#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistical_performance/Convergence_analysis.py;

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

import Cluster_Ensembles as CE       

from itertools import combinations_with_replacement, permutations
import munkres                       # Hungarian or Kuhn-Munkres algorithm
                                     # for the maximum weight bipartite 
                                     # graph matching problem
import numpy as np
from sys import maxint
import tables
from tempfile import NamedTemporaryFile


__all__ = ['convergence_analysis', 'solve_assignment_problem', 
           'test_convergence']


def get_partition_space(N_samples, N_clusters):

    possible_cluster_labels = np.empty((0, N_samples), dtype = int)
    for indices in permutations(np.arange(N_clusters)):
        for label in combinations_with_replacement(indices, N_samples):
            label = list(label)
            if np.unique(label).size == N_clusters:
                possible_cluster_labels = np.vstack((possible_cluster_labels, label))

    return possible_cluster_labels


def get_cluster_runs(N_samples, N_iterations, possible_cluster_labels):

    random_choices = np.random.choice(possible_cluster_labels.shape[0], 
                                      N_iterations)

    cluster_runs = np.empty((N_iterations, N_samples), dtype = float)    
    for i in xrange(N_iterations):
        cluster_runs[i] = possible_cluster_labels[random_choices[i]]

    return cluster_runs


def get_cc_labels(N_samples, N_clusters, N_iterations, N_comparisons):

    possible_cluster_labels = get_partition_space(N_samples, N_clusters)

    cc_labels = np.empty((N_comparisons, N_samples), dtype = int)
    for i in xrange(N_comparisons):
        cluster_runs = get_cluster_runs(N_samples, N_iterations, possible_cluster_labels)
  
        with NamedTemporaryFile('w', suffix = '.h5', delete = True, dir = './') as f:
            fileh = tables.open_file(f.name, 'w')
            fileh.create_group(fileh.root, 'consensus_group')
            fileh.close()

            cc_labels[i] = CE.cluster_ensembles(cluster_runs, f.name, 
                                          N_clusters_max = N_clusters)

    return cc_labels


def solve_assignment_problem(old_labels_matrix):

    N_comparisons, N_samples = old_labels_matrix.shape
    N_clusters = np.amax(old_labels_matrix[0]) + 1

    ref_ind = np.random.choice(N_comparisons)

    new_labels_matrix = np.zeros(old_labels_matrix.shape, dtype = int)
    new_labels_matrix[ref_ind] = old_labels_matrix[ref_ind]

    ref_cc_labels = old_labels_matrix[ref_ind]
    for i in xrange(N_comparisons):
        if i != ref_ind:
            cc_labels_i = old_labels_matrix[i]

            cost_matrix = np.zeros((N_clusters, N_clusters), dtype = int)
            for j in xrange(N_samples):
                cost_matrix[cc_labels_i[j], ref_cc_labels[j]] += 1

            mx = cost_matrix.max()
            cost_matrix = mx - cost_matrix
            # The Hungarian algorithm pertains to minimizing a cost matrix.
            # We are seeking to maximize the cost matrix as initially
            # defined by the assignment problem; this amounts to minimizing
            # the cost matrix whose entry (i,j) is the difference
            # between the maximum value of the original cost matrix and the
            # cost originally at entry (i,j).

            if N_clusters < 6:
                min_cost = maxint
                for p in list(permutations(np.arange(N_clusters))):
                    cost = np.sum(cost_matrix[p, np.arange(N_clusters)])
                    if cost < min_cost:
                        min_cost = cost
                        indices = p
                indices = {i:indices[i] for i in xrange(len(indices))}
            else:
                m = munkres.Munkres()
                indices = dict(m.compute(cost_matrix))
     
            def _f(i):
                assert isinstance(i, int) and 0 <= i < N_clusters
                return indices[i]

            new_labels_matrix[i] = map(_f, cc_labels_i)

    return new_labels_matrix


def test_convergence(f, N_samples, N_clusters, N_iterations, N_comparisons):

    assert isinstance(N_comparisons, int) and N_comparisons > 1

    cc_labels = get_cc_labels(N_samples, N_clusters, N_iterations, N_comparisons)
    cc_labels = solve_assignment_problem(cc_labels)

    f.write('\nN_iterations = {}'.format(N_iterations))
    f.write('\nVectors of consensus cluster IDs:\n')
    np.savetxt(f, cc_labels, fmt = '%d', delimiter = ' ')

    identical_counter = 0
    for i in xrange(N_comparisons):
        for j in xrange(N_comparisons):
            if j != i and np.array_equal(cc_labels[i], cc_labels[j]):
                identical_counter += 1

    f.write('\n{} pairs of cluster IDs are identical, out of {}\n'.format(identical_counter, N_comparisons * (N_comparisons - 1)))


def convergence_analysis():

    N_comparisons = 50

    with open('test_convergence.txt', 'w') as f:
        for N_iter in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]:
            test_convergence(f, 21, 3, N_iter, N_comparisons)
            
