#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistical_performance/Class_tree_edges.py;

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

from collections import OrderedDict
from fractions import gcd
import functools
from itertools import combinations
from math import ceil, floor, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
from sys import exit
import tables
from tempfile import NamedTemporaryFile


__all__ = ['tree_edges']


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
        print("\nERROR: ECLAIR: Statistical_performance: Class_tree_edges: "
              "get_chunk_size: this machine does not have enough free memory "
              "resources to perform ensemble clustering.\n")
        exit(1)


def counter(_lambda):

    def _wrapper(f):

        @functools.wraps(f)
        def _wrapped(self, *args, **kwargs):
            if callable(_lambda):
                ID = _lambda(self)
                if _wrapped.ID == ID:
                    _wrapped.count += 1
                else:
                    _wrapped.ID = ID
                    _wrapped.count = 1
                f(self, *args, **kwargs)
                    
        _wrapped.count = 0
        _wrapped.ID = None

        return _wrapped

    return _wrapper 


class tree_edges(object):

    def __init__(self, ref_tree, max_ref_dist, max_overall_dist):

        self.ref_tree = ref_tree
        self.max_ref_dist = max_ref_dist
        self.max_overall_dist = max_overall_dist

        fh = NamedTemporaryFile('w', suffix = '.h5', delete = True, dir = './')
        self.__fh = fh
        self.__hdf5_storage = fh.name

    def __enter__(self):
   
        return self

    @property
    def ref_tree(self):

        return self.__ref_tree

    @ref_tree.setter
    def ref_tree(self, ref_tree):

        assert isinstance(ref_tree, str), "'ref_tree' must be a name tag, indicating which folder of 'ECLAIR_ensemble_clustering_files' to use"

        try:
            os.path.isdir(ref_tree)
        except EnvironmentError:
            print("'ref_tree' is not a valid directory name")

        self.__ref_tree = ref_tree

    @property
    def max_ref_dist(self):

        return self.__max_ref_dist

    @max_ref_dist.setter
    def max_ref_dist(self, max_ref_dist):

        assert isinstance(max_ref_dist, int) and max_ref_dist > 0, "'max_ref_dist' must be a positive integer"
        self.__max_ref_dist = max_ref_dist

    @property
    def max_overall_dist(self):

        return self.__max_overall_dist

    @max_overall_dist.setter
    def max_overall_dist(self, max_overall_dist):

        assert isinstance(max_overall_dist, int) and max_overall_dist > 0, "'max_ref_dist' must be a positive integer"
        assert max_overall_dist >= self.__max_ref_dist, "'max_overall_dist' must be larger than 'max_ref_dist'"
        self.__max_overall_dist = max_overall_dist

    @counter(lambda x: x.__ref_tree)
    def create_tree_edges_distributions_storage(self):

        fileh = tables.open_file(self.__hdf5_storage, 'w')

        atom = tables.Int64Atom()
        edge_distances_distributions = fileh.create_carray(fileh.root, 'edge_distances_distributions', atom, (self.__max_ref_dist, self.__max_overall_dist + 1), '', filters = None)

        fileh.close()

    @counter(lambda x: x.__ref_tree)
    def wipe_storage(self):

        self.create_tree_edges_distributions_storage()
        
    @staticmethod
    def best_grid(N):

        n = sqrt(N)
        a = int(floor(n))
        b = int(ceil(n))

        possibilities = [a ** 2, a * b, b ** 2]
        min_value = max(possibilities)
        best_i = None
        for i in xrange(3):
            if N <= possibilities[i]:
                if possibilities[i] < min_value:
                    min_value = possibilities[i]
                    best_i = i

        if best_i == 0:
            return (a, a)
        elif best_i == 1:
            return (a, b) if a >= b else (b, a)
        return (b, b)

    def plot(self, output_directory, tag = ''):

        edge_distances_folder = output_directory + '/edge_distances_distributions_wrt_trees_{}'.format(tag)
        try:
            os.makedirs(edge_distances_folder)
        except EnvironmentError:
            if not os.path.isdir(edge_distances_folder):
                print('\nrobustness_metrics\t ERROR\n')
                raise

        fileh = tables.open_file(self.__hdf5_storage, 'r+')

        edge_distances_distributions = fileh.root.edge_distances_distributions
       
        a, b = self.best_grid(self.__max_ref_dist)
        
        fig = plt.figure(1, (2 * a + 5, 2 * b + 5))

        gs = gridspec.GridSpec(a, b)

        edge_list = self.__select_row_indices.keys()

        edge_distances_std = np.zeros(len(edge_list), float)
        for i in xrange(len(edge_list)):
            M = edge_distances_distributions[i]
            M = np.trim_zeros(M, 'b')

            if M.size == 0:
                edge_distances_std[i] = np.nan
                continue

            overall_gcd = reduce(gcd, M)
            M /= overall_gcd

            distances = np.arange(M.size)

            hist = np.divide(M, M.sum() + 0.0)
            bin_edges = np.arange(M.size + 1)
            width = 0.7 * (bin_edges[1] - bin_edges[0])
            center = (bin_edges[:-1] + bin_edges[1:]) / 2

            x, y = divmod(i, b)
            ax = plt.subplot(gs[x, y])
            ax.bar(center, hist, align = 'center', width = width)
    
            ax.set_xticks(center)
            ax.set_xticklabels([str(d) for d in distances])
            ax.set_yticks(np.arange(0, round(np.amax(hist) + 0.1, 1), 0.1))

            ax.set_title('Dispersion across trees of the cells connected\nby edge {} of the reference tree'.format(edge_list[i], self.__ref_tree))

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(6)

            fig.add_subplot(ax)

            mean = np.inner(hist, distances)
            var = np.inner(hist, (distances - mean) ** 2)
            edge_distances_std[i] = sqrt(var)

            with open(edge_distances_folder + '/edge_distances_distribution_{}_{}.tsv'.format(*edge_list[i]), 'w') as f:
                np.savetxt(f, np.vstack((distances, hist)), fmt = '%.6f', delimiter = '\t')

        plt.savefig(output_directory + '/edge_distances_distributions_wrt_trees_{}.pdf'.format(tag))
        plt.close(fig)

        with open(output_directory + '/edge_distances_std_{}.tsv'.format(tag), 'w') as f:
            np.savetxt(f, np.vstack((edge_distances_std, zip(*edge_list))), fmt = '%.6f', delimiter = '\t')

        fileh.close()

    @counter(lambda x: x.__ref_tree)
    def select_row_indices(self):

        with open(self.__ref_tree + '/consensus_adjacency_matrix.txt', 'r') as f:
            tree_adjacency_matrix = np.loadtxt(f, dtype = float)

        tree_adjacency_matrix = tree_adjacency_matrix.astype(int)       

        N_vertices = tree_adjacency_matrix.shape[0]
        vertices = np.arange(0, N_vertices)

        mapped_indices = OrderedDict()

        c = 0
        for i, nodes_pair in enumerate(combinations(vertices, 2)):
            if tree_adjacency_matrix[nodes_pair] == 1:
                mapped_indices[nodes_pair] = i
                c += 1

        assert c == len(mapped_indices)

        self.__select_row_indices = mapped_indices

    @counter(lambda x: x.__ref_tree)
    def update_tree_edges_distributions(self, other_tree, contingency_table_file):

        f_name = other_tree + '/consensus_topological_distances_matrix.txt'
        try:
            os.path.exists(f_name)
        except EnvironmentError:
            print("'other_tree' is not a valid directory name")
        
        with open(f_name, 'r') as f:
            other_tree_dist_mat = np.loadtxt(f, dtype = float)
        
        other_tree_dist_mat = other_tree_dist_mat.astype(int)

        N_other_tree_nodes = other_tree_dist_mat.shape[0]

        iu = np.triu_indices(N_other_tree_nodes, k = 1)
        other_tree_dist = other_tree_dist_mat[iu]
        other_tree_dist = np.append(other_tree_dist, np.zeros(N_other_tree_nodes, dtype = int))

        if self.create_tree_edges_distributions_storage.ID is not self.__ref_tree:
            self.create_tree_edges_distributions_storage()
            # this ensures we create the HDF5 storage structure only once
            # per class instance

        if self.select_row_indices.ID is not self.__ref_tree:
            self.select_row_indices()
            # this makes sure we do not waste time recomputing 
            # those indices again and again

        f_out = tables.open_file(self.__hdf5_storage, 'r+')
        edge_distances_distributions = f_out.root.edge_distances_distributions

        f_in = tables.open_file(contingency_table_file, 'r+')
        contingency_table = f_in.root.inter_trees_group.contingency_table

        row_indices = self.__select_row_indices.values()
        row_indices = np.array(row_indices, dtype = int, copy = False)

        chunks_size = get_chunk_size(contingency_table.shape[1], 4)
        for i in xrange(0, row_indices.size, chunks_size):
            max_ind = min(i + chunks_size, row_indices.size)

            chunk_row_indices = row_indices[i:max_ind]
            M = contingency_table[chunk_row_indices, :]

            N = np.zeros((max_ind - i, self.__max_overall_dist + 1), dtype = int)
            ixgrid = np.ix_(np.arange(max_ind - i), other_tree_dist)

            np.add.at(N, ixgrid, M)
            # the line above is meant to ensure that results are summed 
            # for elements that are indexed more than once
            edge_distances_distributions[i:max_ind] += N

            if self.update_tree_edges_distributions.count == 1 or (self.wipe_storage.count > 0 and self.wipe_storage.ID is self.__ref_tree):
                edge_distances_distributions[i:max_ind, 1] += M.sum(axis = 1)

        f_in.close()
        f_out.close()   

    def __exit__(self, type, value, traceback):

        self.__fh.close()
        
