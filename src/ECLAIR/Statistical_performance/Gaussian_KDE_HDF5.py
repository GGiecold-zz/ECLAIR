#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistical_performance/Gaussian_KDE_HDF5.py;

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

from math import pi, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import scipy._lib.six
from sys import exit
import tables


__all__ = ['gaussian_kde']


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
        print("\nERROR: ECLAIR: Statistical_performance: Gaussian_KDE_HDF5: "
              "get_chunk_size: this machine does not have enough free memory "
              "resources to perform ensemble clustering.\n")
        exit(1)


class gaussian_kde(object):

    def __init__(self, data_hdf5_storage, bw_method = None):

        self.__hdf5_file = data_hdf5_storage

        with tables.open_file(self.__hdf5_file, 'r+') as fileh:
            dataset = fileh.root.dataset
            
            self.__N_features, self.__N_training = dataset.shape

            if not self.__N_features * self.__N_training > 1:
                raise ValueError("'dataset' should have more than one entry")
            
            if hasattr(fileh.root, 'repetitions'):
                assert (1, self.__N_training) == fileh.root.repetitions.shape, 'The list of repeated values should have the same length as the list of values itself.'
                self.repetitions_flag = True
                
                repetitions = fileh.root.repetitions
                
                szum = 0
                chunks_size = get_chunk_size(1, 1)
                for i in xrange(0, self.__N_training, chunks_size):
                    max_ind = min(i + chunks_size, self.__N_training)
                    szum += np.sum(repetitions[0, i:max_ind])
                
                self.__N_training_samples = szum
            else:
                self.repetitions_flag = False
                self.__N_training_samples = self.__N_training

        self.set_bandwidth(bw_method = bw_method)

    def evaluate(self, eval_set):

        eval_set = np.atleast_2d(eval_set)

        N_features, N_eval = eval_set.shape

        if N_features != self.__N_features:
            if d == 1 and N_eval == self.__N_features:
                eval_set = eval_set.reshape((self.__N_features, 1))
                N_eval = 1
            else:
                raise ValueError('The test subsample provided has dimension {}, whereas the training dataset is {}-dimensional.'.format(N_features, self.__N_features))

        values = np.zeros((N_eval, ), dtype = float)

        with tables.open_file(self.__hdf5_file, 'r+') as fileh:
            dataset = fileh.root.dataset
            
            if self.repetitions_flag:
                repetitions = fileh.root.repetitions

            for i in xrange(N_eval):
                eval_point = eval_set[:, i, np.newaxis]

                chunks_size = get_chunk_size(self.__N_features, 4)
                for j in xrange(0, self.__N_training, chunks_size):
                    max_ind = min(j + chunks_size, self.__N_training)
        
                    M = dataset[:, j:max_ind]

                    diff = M - eval_point
                    tdiff = np.dot(self.__inverse_covariance, diff)

                    energy = np.sum(diff * tdiff, axis = 0) / 2.0
                
                    if self.repetitions_flag:
                        R = repetitions[0, j:max_ind]
                        
                        values[i] = np.sum(R * np.exp(-energy), axis = 0)
                    else:
                        values[i] = np.sum(np.exp(-energy), axis = 0)

        values /= float(self.__normalization_factor)

        return values

    __call__ = evaluate

    def scott_factor(self):

        return np.power(self.__N_training_samples / float(4000), - 1 / (self.__N_features + 4.0))

    def silverman_factor(self):

        return np.power(self.__N_training_samples * (self.__N_features + 2.0) / 4.0, - 1 / (self.__N_features + 4.0))

    _covariance_factor = scott_factor
    # default method to calculate the bandwidth

    def set_bandwidth(self, bw_method = None):

        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.__covariance_factor = self.scott_factor
        elif bw_method == 'silverman':
            self.__covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, scipy._lib.six.string_types):
            self.__bw_method = 'use constant'
            self.__covariance_factor = lambda: bw_method
        elif scipy._lib.six.callable(bw_method):
            self.__bw_method = bw_method
            self.__covariance_factor = lambda: self._bw_method(self)
        else:
            raise ValueError("'bw_method' should be set to either of 'scott', 'silverman', a scalar or a callable.")

        self._get_inverse_covariance()

    def _get_inverse_covariance(self):

        self.factor = self._covariance_factor()

        if not hasattr(self, '_data_inverse_covariance'):
            self._get_means_and_variances()
            self._get_covariance()
            self.__data_inverse_covariance = np.linalg.inv(self.__data_covariance)
        
        self.__covariance = self.__data_covariance * (self.factor ** 2)
        self.__inverse_covariance = np.divide(self.__data_inverse_covariance, (self.factor ** 2))

        eensy = np.finfo(np.float32).eps

        self.__normalization_factor = sqrt(np.linalg.det(2 * pi * self.__covariance) + eensy) * self.__N_training_samples

    def _get_means_and_variances(self):

        with tables.open_file(self.__hdf5_file, 'r+') as fileh:
            dataset = fileh.root.dataset
            
            if self.repetitions_flag:
                repetitions = fileh.root.repetitions

            means = np.zeros(self.__N_features, dtype = float)
            variances = np.zeros(self.__N_features, dtype = float)

            smallest = dataset[:, 0]
            largest = dataset[:, 0]

            chunks_size = get_chunk_size(self.__N_features, 4)
            for i in xrange(0, self.__N_training, chunks_size):
                max_ind = min(i + chunks_size, self.__N_training)

                M = dataset[:, i:max_ind]

                if self.repetitions_flag:
                    R = repetitions[0, i:max_ind]
                    
                    T = M * R        
                    means += T.sum(axis = 1)
                    
                    T = np.square(M) * R
                    variances += T.sum(axis = 1)
                else:
                    means += M.sum(axis = 1)
                    variances += np.square(M).sum(axis = 1)

                smallest = np.minimum(smallest, np.amin(M, axis = 1))
                largest = np.maximum(largest, np.amax(M, axis = 1))

        self.__mins = smallest
        self.__maxs = largest

        means /= float(self.__N_training_samples)
        variances /= float(self.__N_training_samples - 1)
        variances -= (self.__N_training_samples / float(self.__N_training_samples - 1)) * np.square(means)

        self.__means = means
        self.__variances = variances

    def _get_covariance(self):

        with tables.open_file(self.__hdf5_file, 'r+') as fileh:
            dataset = fileh.root.dataset
            
            if self.repetitions_flag:
                repetitions = fileh.root.repetitions

            covariance_matrix = np.zeros((self.__N_features, self.__N_features), dtype = float)

            chunks_size = get_chunk_size(self.__N_features, 4)
            for i in xrange(0, self.__N_training, chunks_size):
                max_ind = min(i + chunks_size, self.__N_training)

                M = dataset[:, i:max_ind]
                M -= self.__means[:, np.newaxis]
                
                if self.repetitions_flag:
                    R = repetitions[0, i:max_ind]
            
                for j in xrange(self.__N_features - 1):
                    M_j = M[j]
                    for k in xrange(j + 1, self.__N_features):
                        if self.repetitions_flag:
                            x = M[k] * R
                            x = np.inner(M_j, x)
                        else:
                            x = np.inner(M_j, M[k]) 
                        
                        covariance_matrix[j, k] += x
                        covariance_matrix[k, j] += x

        covariance_matrix /= float(self.__N_training_samples - 1)
        covariance_matrix[np.diag_indices(self.__N_features)] = self.__variances

        self.__data_covariance = covariance_matrix

    def __getattr__(self, name):
    
        if name == 'smallest':
            return self.__mins
        elif name == 'largest':
            return self.__maxs    
        elif name == 'covariance_matrix':
            return self.__data_covariance
    
        class_name = self.__class__.__name__
    
        if name in frozenset({'hdf5_file', 'N_features', 'N_training', 'inverse_covariance', 'normalization_factor', 'covariance_factor', 'bw_method', 'means', 'variances'}):
            return self.__dict__['_{class_name}__{name}'.format(**locals())]
            
        raise AttributeError("'{class_name}' object has no attribute '{name}'".format(**locals()))

    def pdf(self, eval_set):

        return self.evaluate(eval_set)

    def logpdf(self, eval_set):

        return np.log(self.evaluate(eval_set))
        
    def plot(self, output_directory = '.', tag = '', N_bins = 100):
    
        assert isinstance(N_bins, int) and N_bins > 1

        mins = self.__mins
        maxs = self.__maxs

        x_i, y_i = np.mgrid[mins[0]:maxs[0]:N_bins*1j, mins[1]:maxs[1]:N_bins*1j]
        z_i = self.evaluate(np.vstack((x_i.flatten(), y_i.flatten())))

        fig = plt.figure()
        fig.suptitle("Gaussian KDE of the pairwise distances between cells.",
                     fontsize = 10)

        plt.pcolormesh(x_i, y_i, z_i.reshape(x_i.shape), rasterized = True)

        fig.tight_layout()
        plt.savefig(output_directory + '/Gaussian_kde_inter_trees_pairwise_cells_distances_{}.pdf'.format(tag))
        plt.close(fig)

        try:
            os.makedirs(output_directory + '/KDE_matrices_{}'.format(tag))
        except OSError:
            if not os.path.isdir(output_directory + '/KDE_matrices_{}'.format(tag)):
                raise

        with open(output_directory+'/KDE_matrices_{}/xyz.txt'.format(tag), 'w') as f:
            np.savetxt(f, np.vstack((x_i.flatten(), y_i.flatten(), z_i)), 
                       fmt = '%.6f')


