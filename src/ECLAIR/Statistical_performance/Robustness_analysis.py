#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistics/Robustness_analysis.py;

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

from ..Build_instance import ECLAIR_core as ECLR
from .Statistical_tests import robustness_metrics

from collections import namedtuple
import numpy as np
import operator
import os
import pkg_resources
import random
from sklearn import cross_validation
from sklearn.metrics import pairwise_distances_argmin_min
from tempfile import NamedTemporaryFile
import tarfile
import time
import zipfile


__all__ = ['ECLAIR_generator', 'experiment_1', 'experiment_2', 'experiment_3']


def extract_file(path, output_directory = '.'):

    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else: 
        raise ValueError, "\nERROR: ECLAIR: Robustness_analysis: failed to extract {0}; no appropriate extractor could be found".format(path)
    
    cwd = os.getcwd()
    os.chdir(output_directory)
    
    try:
        file = opener(path, mode)
        try: 
            file.extractall()
        finally: 
            file.close()
    finally:
        os.chdir(cwd)
    

def ECLAIR_generator(data_file_name, sampling_fraction, N_runs, N_iter, 
                     method, k, output_directory, data_flag = 'CyTOF'):
    """Automatically runs the ECLAIR method on a dataset
       accessible via 'data_file_name' so as to generate 'N_iter'
       independent consensus clusterings and associated minimum spanning
       trees. 

    Parameters
    ----------
    data_file_name : file object or string
        A path to the dataset under consideration. 
        Any dataset can be submitted to this routine,
        with the proviso that it has previously been
        mangled to meet a few constraints regarding
        headers, delimiters, etc.
        Those constraints are handled hereby
        for a qPCR dataset and an aggregation
        of flow cytometry mouse bone marrow samples.

    sampling_fraction : float
        Specifies the number of points to downsample from the
        dataset considered a each of 'N_iter' stages,
        before applying k-means clustering
        to this group obtained via a density-based approach.

    k : int
        The parameter used for k-means clustering each 
        downsampled group of data-points, as required
        for all 'N_runs' intermediate steps of ECLAIR.

    N_runs : int
        The number of independent runs of downsampling and clustering
        to perform before applying our ensemble clustering algorithm
        to this group.

    N_iter : int
        The number of ensemble clusterings and accompanying trees 
        to generate by k-fold cross validation, with k = N_iter.
        We randomly reshuffle the dataset and split it into
        N_iter equally-sized parts. Of the N_iter subsamples,
        a single subsample is kept as a 'validation data', while
        the other serve as the 'training data' from which we build
        an ensemble clustering and afferent minimum spanning tree.
        Only upon obtaining this ensemble clustering, do we
        ascribe each data point from the left-over 'validation' subsample
        to its nearest cluster in gene expression space.
        Each sample from the whole dataset therefore has a cluster label.
        Such vectors of cluster identities are then used in other
        functions of this module for various comparisons between trees
        and consensus clusterings.

    output_directory : file object or string
        The path to the folder where the information and figures
        associated with each of 'N_iter' rounds of consensus clustering
        are to be stored.

    test_set_flag : bool, optional (default = False)

    data_flag : string, optional (default = 'CyTOF')
        Allows the processing of a 'CyTOF' dataset
        (Supplementary dataset 2 from 
         Qiu et al., Nature Biotechnology, Vol. 29, 10 (2011))

    Returns
    -------
    name_tags : list
        Within 'output_directory', records the names of the folder 
        associated to each of 'N_iter' consensus clusterings obtained. 
    """

    assert method in {'hierarchical', 'k-means'}

    assert data_flag in {'CyTOF', 'qPCR'}
    # Our method has been thoroughly tested on the two corresponding datasets.
    # Unlike the preceding procedures, 'ECLAIR_generator' is akin to a script 
    # due to all the peculiarities in the number of features kept 
    # for downstream analysis, separators, etc.

    if data_flag == 'CyTOF':
        skiprows = 1
        delimiter = '\t'
        usecols = [3, 4, 5, 7, 8, 9, 10, 12, 13]
    elif data_flag == 'qPCR':
        skiprows = 1
        delimiter = '\t'
        usecols = xrange(1, 49)
    # keeping open the addition of other datasets 
    # to be submitted to the present routine

    with open(data_file_name, 'r') as f: 
        data = np.loadtxt(f, dtype = float, skiprows = skiprows, 
                          delimiter = delimiter, usecols = usecols)
        # in the case of the CyTOF mouse bone marrow experiment, 
        # load the samples resulting from an arcSinh transformation
        # applied to the raw dataset

    if method == 'hierarchical':
        HIERARCHICAL_parameters = namedtuple('HIERARCHICAL_parameters',
                                             'clustering_method k')
        clustering_parameters = HIERARCHICAL_parameters('hierarchical', k)
    elif method == 'k-means':
        KMEANS_parameters = namedtuple('KMEANS_parameters', 'clustering_method k')
        clustering_parameters = KMEANS_parameters('k-means', k)
    # leaving open the extension of this analysis to other clustering methods

    CC_parameters = namedtuple('CC_parameters', 'N_runs sampling_fraction N_cc')
    cc_parameters = CC_parameters(N_runs, sampling_fraction, k)

    try:
        os.makedirs(output_directory)
    except OSError:
        if not os.path.isdir(output_directory):
            print('\nECLAIR_generator\t ERROR\n')
            raise

    N_samples = data.shape[0]

    # separate the samples into 'N_iter' groups of equal size, 
    # by random selection with no replacement:
    kf = cross_validation.KFold(N_samples, n_folds = N_iter, shuffle = True)

    name_tags = []

    c = 1
    for test_indices, train_indices in kf:
        training_data = np.take(data, train_indices, axis = 0)
        if data_flag == 'CyTOF':
            # replacing by cell IDs the column keeping 
            # track of measurement times:
            training_data[:, 0] = np.arange(train_indices.size)

        train_indices = train_indices.reshape((1, train_indices.size))
        with open(output_directory + '/training_{}.txt'.format(c), 'w') as f:
            np.savetxt(f, train_indices, fmt = '%d', delimiter = '\t', newline = '\n')

        with open(output_directory + '/training_data_{}.tsv'.format(c), 'w') as f:
            np.savetxt(f, training_data, fmt = '%.6f', delimiter = '\t')

        Data_info = namedtuple('Data_info', 'data_file_name expected_N_samples skip_rows cell_IDs_column extra_excluded_columns time_info_column')
        data_info = Data_info(output_directory + '/training_data_{}.tsv'.format(c),
                              train_indices.size, 0, 0, None, -1)

        with NamedTemporaryFile('w', suffix = '.h5', delete = True, dir = './') as f:
            name_tag = ECLR.ECLAIR_processing(f.name, data_info,
                           clustering_parameters, cc_parameters, output_directory)
            name_tags.append(name_tag)

        cluster_IDs_file = output_directory + '/ECLAIR_ensemble_clustering_files/' + str(name_tags[-1]) + '/consensus_labels.txt'
        with open(cluster_IDs_file, 'r') as f:
            cluster_IDs = np.loadtxt(f, dtype = int)

        method = clustering_parameters.clustering_method

        cluster_IDs = upsample(test_indices, cluster_IDs, data, method, 
                               xrange(1, data.shape[1]))

        os.remove(cluster_IDs_file)

        with open(cluster_IDs_file, 'w') as f:
            np.savetxt(f, cluster_IDs, fmt = '%d', delimiter = '\t')
    
        c += 1

    return name_tags


def upsample(test_indices, training_set_cluster_IDs, data, 
             method = 'k-means', usecols = None):

    N_samples = test_indices.size + training_set_cluster_IDs.size

    assert N_samples == data.shape[0]

    full_set_cluster_IDs = np.zeros(N_samples, dtype = int)

    training_indices = np.setdiff1d(np.arange(N_samples), test_indices, True)
    full_set_cluster_IDs[training_indices] = training_set_cluster_IDs

    if usecols is not None:
        usecols = list(usecols)
        data = np.take(data, usecols, 1)    

    training_data = np.delete(data, test_indices, axis = 0)
    
    max_ID = np.amax(training_set_cluster_IDs)
    centroids = np.zeros((max_ID + 1, data.shape[1]), dtype = float)

    for cluster in xrange(max_ID + 1):
        samples_in_cluster = np.where(training_set_cluster_IDs == cluster)[0]
        if method == 'hierarchical':
            centroids[cluster] = np.median(training_data[samples_in_cluster], 
                                           axis = 0)
        else:
            centroids[cluster] = training_data[samples_in_cluster].mean(axis = 0)

    test_data = np.take(data, test_indices, axis = 0)
    test_set_cluster_IDs, _ = pairwise_distances_argmin_min(test_data, centroids, 
                metric = 'manhattan' if method == 'hierarchical' else 'euclidean')

    full_set_cluster_IDs[test_indices] = test_set_cluster_IDs

    return full_set_cluster_IDs


def experiment_1(N_iter, data_flags, method = 'k-means', test_set_flag = True):
    """
    Parameters:
    -----------

    N_iter : int
        Number of replicate experiments to generate
    """
 
    assert not reduce(operator.and_, data_flags)
    assert reduce(operator.xor, data_flags)

    assert isinstance(N_iter, int) and N_iter > 1
    
    try:
        os.makedirs('./ECLAIR_performance')
    except OSError:
        if not os.path.isdir('./ECLAIR_performance'):
            print('\nERROR: ECLAIR: Robustness_analysis: experiment_1\n')
            raise

    start_t = time.time()

    ECLAIR_qPCR_flag, ECLAIR_CyTOF_flag, SPADE_CyTOF_flag = data_flags

    if ECLAIR_CyTOF_flag:
        output_directory = './ECLAIR_performance/ECLAIR_test_sets_CyTOF'
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                print('\nERROR: ECLAIR: Robustness_analysis: experiment_1\n')
                raise
        
        # Access path to the CyTOF mouse bone marrow dataset
        compressed_data_path = pkg_resources.resource_filename(__name__,
                        'data/SPADE_data/nbt-SD2-Transformed.tsv.tar.gz')
        extract_file(compressed_data_path, './ECLAIR_performance')
        data_file = './ECLAIR_performance/nbt-SD2-Transformed.tsv'

        max_N_clusters = 50

        name_tags = ECLAIR_generator(data_file, 0.5, 100, N_iter, method, max_N_clusters, output_directory)

        _ = robustness_metrics(max_N_clusters, output_directory + '/ECLAIR_ensemble_clustering_files', name_tags, 
                               output_directory, test_set_flag)

        _ = robustness_metrics(max_N_clusters, output_directory + '/ECLAIR_ensemble_clustering_files', name_tags, 
                               output_directory, test_set_flag, MST_flag = False)

    elif ECLAIR_qPCR_flag:
        data_file = pkg_resources.resource_filename(__name__,
                        'data/Guoji_data/qPCR.txt')                                 
        output_directory = './ECLAIR_performance/ECLAIR_test_sets_qPCR'
        
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                print('\nERROR: ECLAIR: Robustness_analysis: experiment_1\n')
                raise

        max_N_clusters = 13

        name_tags = ECLAIR_generator(data_file, 0.2, 50, N_iter, method,
                                     max_N_clusters, output_directory, 'qPCR')

        _ = robustness_metrics(max_N_clusters, output_directory + '/ECLAIR_ensemble_clustering_files', name_tags, 
                               output_directory, test_set_flag)

        _ = robustness_metrics(max_N_clusters, output_directory + '/ECLAIR_ensemble_clustering_files', name_tags, 
                               output_directory, test_set_flag, MST_flag = False)

    elif SPADE_CyTOF_flag:
        max_N_clusters = 50
        
        output_directory = './ECLAIR_performance/SPADE_test_sets_CyTOF'
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                print('\nERROR: ECLAIR: Robustness_analysis: experiment_1\n')
                raise
            
        SPADE_files = pkg_resources.resource_filename(__name__,
                          'data/SPADE_test_sets')
        
        for i in xrange(1, 4):                  
            with open(SPADE_files + '/training_{0}.txt'.format(i), 'r') as f:
                training_set = np.loadtxt(f, dtype = int, delimiter = '\t')
                
            with open(output_directory + '/training_{0}.txt'.format(i), 'w') as f:
                np.savetxt(f, training_set, fmt = '%d', delimiter = '\t')
                      
        name_tags = ['training_1', 'training_2', 'training_3']

        _ = robustness_metrics(max_N_clusters, SPADE_files, name_tags, 
                               output_directory, test_set_flag)

    end_t = time.time()

    print('\n{}_robustness\t SUMMARY\t:\nthe whole process of comparing those minimum-spanning trees and the associated consensus clusterings took {} seconds.\n'.format('SPADE' if SPADE_CyTOF_flag else 'ECLAIR', round(end_t - start_t, 2)))


def experiment_2(data_file_name, k, sampling_fraction = 0.2, N_runs = 50):

    output_directory = './ECLAIR_performance/ECLAIR_same_dataset'
    try:
        os.makedirs(output_directory)
    except OSError:
        raise

    with open(data_file_name, 'r') as f: 
        data = np.loadtxt(f, dtype = float, skiprows = 1, delimiter = '\t')

    N_samples = data.shape[0]

    for i in xrange(1, 51):
        with open(output_directory + '/training_{}.txt'.format(i), 'w') as f:
            np.savetxt(f, np.arange(N_samples), fmt = '%d')

    KMEANS_parameters = namedtuple('KMEANS_parameters', 'clustering_method k')
    clustering_parameters = KMEANS_parameters('k-means', k)

    CC_parameters = namedtuple('CC_parameters', 'N_runs sampling_fraction N_cc')
    cc_parameters = CC_parameters(N_runs, sampling_fraction, k)    

    Data_info = namedtuple('Data_info', 'data_file_name expected_N_samples skip_rows cell_IDs_column extra_excluded_columns time_info_column')
    data_info = Data_info(data_file_name, N_samples, 1, 0, None, -1)

    name_tags = []

    for i in xrange(50):
        with NamedTemporaryFile('w', suffix = '.h5', delete = True, dir = './') as f:
            name_tag = ECLR.ECLAIR_processing(f.name, data_info,
                           clustering_parameters, cc_parameters,
                           output_directory)
            name_tags.append(name_tag)

    _ = robustness_metrics(k, output_directory + '/ECLAIR_ensemble_clustering_files',
                           name_tags, output_directory, test_set_flag = False)

    _ = robustness_metrics(k, output_directory + '/ECLAIR_ensemble_clustering_files',
                           name_tags, output_directory, test_set_flag = False,
                           MST_flag = False)


def experiment_3():

    output_directory = './ECLAIR_performance/SPADE_same_dataset'
    try:
        os.makedirs(output_directory)
    except OSError:
        if not os.path.isdir(output_directory):
            print('\nERROR: ECLAIR: Robustness_analysis: experiment_3\n')
            raise

    max_N_clusters = 50
 
    name_tags = ['training_{0}'.format(i) for i in xrange(1, 11)]
    
    SPADE_files = pkg_resources.resource_filename(__name__,
                      'data/SPADE_same_dataset')
                      
    with open(SPADE_files + '/training.txt', 'r') as f:
        training_set = np.loadtxt(f, dtype = int, delimiter = '\t')
            
    for i in xrange(1, 11):
        with open(output_directory + '/training_{0}.txt'.format(i), 'w') as f:
            np.savetxt(f, training_set, fmt = '%d', delimiter = '\t')

    _ = robustness_metrics(max_N_clusters, SPADE_files, name_tags, 
                           output_directory, test_set_flag = False)

