#!/usr/bin/env python


# ECLAIR/ECLAIR/Build_instance/__main__.py


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


from .ECLAIR_core import ECLAIR_processing

from collections import namedtuple
import datetime
import locale
from math import floor, sqrt
import optparse
import os
import psutil
import re
import sys
import tables


Data_info = namedtuple('Data_info', "data_file_name expected_N_samples "
                       "skip_rows cell_IDs_column extra_excluded_columns "
                       "time_info_column")
AP_parameters = namedtuple('AP_parameters', "clustering_method max_iter "
                           "convergence_iter")
DBSCAN_parameters = namedtuple('DBSCAN_parameters', "clustering_method "
                               "minPts eps quantile metric")
HIERARCHICAL_parameters = namedtuple('HIERARCHICAL_parameters', 
                                     'clustering_method k')
KMEANS_parameters = namedtuple('KMEANS_parameters', 'clustering_method k')

CC_parameters = namedtuple('CC_parameters', 'N_runs sampling_fraction N_cc')


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
  

def user_interface():
    """Retrieve the path to the data-file and various related parameters.
       Propose to choose from different clustering algorithms 
       and ask how many runs ('N_runs') of those to perform 
       on random samples of fixed-size ('sampling-fraction').

    Returns
    -------
    data_info : file object or string

    clustering_parameters : namedtuple
 
    cc_parameters : namedtuple
    """

    print('\n*****************************************')
    print('*****************************************')
    print('***             ECLAIR                ***')
    print('*****************************************')
    print('*****************************************\n')

    try:
        data_file_name = raw_input("\nECLAIR: provide the path to the file "
                                   "holding the data to be analyzed:\n")
    except:
        data_file_name = ''
    while not os.path.exists(data_file_name):
        try:
            data_file_name = raw_input("\nECLAIR: this file does not exist. "
                                       "Try again:\n")
        except:
            data_file_name = ''

    try:
        skip_rows = int(input("\nECLAIR: how may rows count as header in this file? "
                              "Enter '0' if the file is not adorned by any "
                              "header:\n"))
    except:
        skip_rows = -1
    while 0 > skip_rows:
        try: 
            skip_rows = int(input("\nECLAIR: please provide a non-negative "
                                  "integer:\n")) 
        except:
            skip_rows = -1

    try:
        cell_IDs_column = int(input("\nECLAIR: which column of the data-file "
                                    "holds the names, tags or IDs of its samples? "
                                    "Enter '0' for the 1st column, '1' for the "
                                    "second, etc.:\n"))
    except:
        cell_IDs_column = -1
    while 0 > cell_IDs_column:
        try:
            cell_IDs_column = int(input('\nECLAIR: invalid entry. Try again:\n'))
        except:
            cell_IDs_column = -1

    answer = str(raw_input("\nECLAIR: does this data-set include some "
                           "time information? [Y/n] \n"))
    if answer in {'y', 'Y', 'yes', 'Yes', 'YES', 'yep', 'Yep', 'YEP'}:
        try:
            time_info_column = int(input("\nECLAIR: provide the column holding "
                                         "that information:\n"))
        except:
            time_info_column = -1
        while 0 > time_info_column:
            try:
                time_info_column = int(input("\nECLAIR: it must be a non-negative "
                                             "number. Try again:\n"))
            except:
                time_info_column = -1
    elif answer in {'n', 'N', 'no', 'No', 'NO', 'nope', 'Nope', 'non', 'Non', 'NON', 'nada', 'Nada', 'NADA'}:
        time_info_column = -1
    else:
        print("\nECLAIR: I'll take this as a 'no'.\n")
        time_info_column = -1

    try:
        s = raw_input("\nECLAIR: you may choose to exclude some columns "
                      "as features. If this option does not apply, simply press "
                      "'Enter'. Otherwise, provide a list of numbers:\n")
        extra_excluded_columns = [int(i) for i in re.findall(r'\d+', s)]
    except:
        extra_excluded_columns = None

    try:
        expected_N_samples = int(input("\nECLAIR: please give an estimate "
                                       "of the number of samples in this "
                                       "data-set:\n"))
    except:
        expected_N_samples = -1
    while 0 >= expected_N_samples:
        try:
            expected_N_samples = int(input('\nPlease provide a positive integer:\n'))
        except:
            expected_N_samples = -1

    data_info = Data_info(data_file_name, expected_N_samples, skip_rows,
                          cell_IDs_column, extra_excluded_columns, 
                          time_info_column)
 
    try:
        N_runs = int(input("\nECLAIR: please enter the number of trees "
                           "that will be bagged into a forest "
                           "(a value of '50' is recommended):\n"))
    except:
        N_runs = -1
    while 0 >= N_runs:
        try: 
            N_runs = int(input('\nECLAIR: please provide a positive integer:\n'))
        except:
            N_runs = -1

    try:
        sampling_fraction = float(input("\nECLAIR: how many points do you want "
                                        "to sample from the dataset? "
                                        "Please provide a fraction of the "
                                        "total number of cells:\n"))
    except:
        sampling_fraction = -1.0
    while sampling_fraction > 1 and sampling_fraction < 0: 
        try:
            sampling_fraction = float(input("\nECLAIR: invalid value; "
                                            "please choose a number in (0;1):\n"))
        except:
            sampling_fraction = -1.0

    print('\nECLAIR: choose the clustering algorithm to be applied to each of {:n} subsamples from your data-set.\nAvailable methods: affinity propagation (1), DBSCAN (2), hierarchical clustering (3) & k-means (4)\n'.format(N_runs))
    while True:
        try:
            method = int(input())
        except:
            method = -1
        if method in {1,2,3, 4}:
            break
        print('\nECLAIR: invalid value; please enter either 1, 2, 3 or 4:\n')

    if method == 1:
        required_space = 16 * (expected_N_samples ** 2)
        ratio = int(floor(memory()['free'] / required_space))
        if ratio >= 5:
            recommended_max_iter = 200
        elif ratio > 0:
            recommended_max_iter = 50 * ratio
        else:
            ratio = int(floor(required_space / memory()['free']))
            recommended_max_iter = int(floor(50 / sqrt(ratio)))

        try:
            max_iter = int(input('\nECLAIR: how many rounds of messages-passing among the data-points do you wish be performed? (Recommended value, based on the size of your data-set and the free memory on your computer: {:n})\n'.format(recommended_max_iter)))
        except:
            max_iter = -1
        while 0 >= max_iter:
            try:
                max_iter = int(input("\nECLAIR: invalid entry; "
                                     "please provide a positive integer:\n"))
            except:
                max_iter = -1

        if max_iter > recommended_max_iter:
            answer = str(raw_input("\nECLAIR: affinity-propagation is quite "
                                   "time-consuming. It also requires matrices of "
                                   "similarities, availabilities and responsibilities "
                                   "whose size possibly exceeds what can fit in memory "
                                   "on this machine. Unlike the version provided "
                                   "in scikit-learn, our implementation of affinity "
                                   "propagation clustering can handle this issue; "
                                   "this adds however to the overall burden.\nAre you "
                                   "sure you want to proceed with this value? [Y/n]\n"))
            if answer in {'n', 'N', 'no', 'No', 'NO', 'nope', 'Nope', 'NOPE'}:
                try:
                    max_iter = int(input("\nECLAIR: wise choice! Please enter "
                                         "a new value for this parameter: "))
                except:
                    max_iter = -1
                while 0 >= max_iter and max_iter > recommended_max_iter:
                    try:
                        max_iter = int(input('\nECLAIR: please provide a positive integer, less than {:n}\n'.format(recommended_max_iter)))
                    except:
                        max_iter = -1 
        
        while True:
            try:
                convergence_iter = int(input("\nECLAIR: after how many iterations "
                                             "with no change in the number of "
                                             "estimated clusters should affinity "
                                             "propagation stop? "
                                             "(Recommended value: 15)\n"))
            except:
                convergence_iter = -1

            if 0 >= convergence_iter:
                print('\nECLAIR: invalid entry; try again!\n')
                continue
            elif convergence_iter > 30:
                print("\nECLAIR: too big! The A.I. is out of the box and taking "
                      "control of this machine, setting this parameter to its "
                      "default value of 15.\n")
                convergence_iter = 15
                break
            else:
                break

        clustering_parameters = AP_parameters('affinity_propagation', max_iter,
                                              convergence_iter)

    elif method == 2:
        print("\nECLAIR: you have chosen to perform a Density-Based Spatial Clustering of Applications with Noise on each of {:n} samples from the data-set. Unless you decide to manually enter a value for the radius 'epsilon', this parameter - which determining density reachability - will be determined automatically upon inspection of the distribution of pairwise distances for your data-set and based on a choice of metric you will be asked to provide.".format(N_runs)
)

        try:
            minPts = int(input("\nECLAIR: how many points are needed to form "
                               "a dense region?\n"))
        except:
            minPts = -1
        while 0 >= minPts:
            try:
                minPts = int(input("\nECLAIR: sorry but 'minPts' must be "
                                   "a positive integer; try again:\n"))
            except:
                minPts = -1

        answer = str(raw_input("\nECLAIR: do you want to provide a value of the parameter 'epsilon' for DBSCAN? [Y/n] \nIf not, as is recommended but might take some time, 'epsilon' will be determined in an adpative way from a {:n}-distance graph.\n".format(minPts)))
        if answer in {'y', 'Y', 'yes', 'Yes', 'YES', 'yep', 'Yep', 'YEP', 'Yoh, man!', 'si', 'Si', 'SI', 'oui', 'Oui', 'OUI', 'da','Da', 'DA'}:
            try:
                eps = float(input("\nECLAIR: please provide a value for "
                                  "'epsilon':\n"))
            except:
                eps = 0.0
            while eps <= 0.0:
                try:
                    eps = float(input("\nECLAIR: you must provide a "
                                      "positive number. Give it another stab:\n"))
                except:
                    eps = 0.0
        elif answer in {'n', 'N', 'no', 'No', 'NO', 'nope', 'Nope', 'non', 'Non', 'NON', 'nada', 'Nada', 'NADA'}:
            eps = None
        else:
            print("\nECLAIR: I'll take this as a 'no'.\n")
            eps = None

        quantile = 50

        if eps is None:
            answer = str(raw_input("\nECLAIR: do you want to specify epsilon as a particular quantile to a distribution of {:n}-nearest distances? Please answer by [Y/n]. If not, epsilon will default to the median of that distribution.\n".format(minPts)))
            if answer in {'y', 'Y', 'yes', 'Yes', 'YES', 'yep', 'Yep', 'YEP','si', 'Si', 'SI', 'oui', 'Oui', 'OUI', 'da','Da', 'DA'}:
                try:
                    quantile = float(input("\nECLAIR: please provide a value "
                                           "for the quantile:\n"))
                except:
                    quantile = -1.0
                while quantile < 0 or quantile > 100:
                    try:
                        quantile = float(input("\nECLAIR: sorry but try again "
                                               "and provide a real number "
                                               "within [0; 100]:\n"))
                    except:
                        quantile = -1.0
        
        valid_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                         'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                         'jaccard', 'kulsinski', 'l1', 'l2', 'mahalanobis',
                         'manhattan', 'matching', 'minkowski', 'rogerstanimoto',
                         'russellrao', 'seuclidean', 'sqeuclidean', 'sokalmichener',
                         'sokalsneath', 'wminkowski', 'yule']

        try:
            metric = str(raw_input("\nECLAIR: metric for calculating the distance "
                                   "between instances in your data-set "
                                   "(default would be 'minkowski'):\n")).lower()
        except:
            metric = ''
        while metric not in valid_metrics:
            try:
                metric = str(raw_input('\nECLAIR: this choice is not recognized. Please pick one from:\n{0}\n'.format(valid_metrics))).lower()
            except:
                metric = ''

        clustering_parameters = DBSCAN_parameters('DBSCAN', minPts, eps, quantile, metric)

    else:
        try:
            n_clusters = int(input('\nECLAIR: how many centroids to generate for each run of {0} clustering?\n'.format('hierarchical' if method == 3 else 'k-means')))
        except:
            n_clusters = -1
        while 0 >= n_clusters:
            try:
                n_clusters = int(input("\nECLAIR: invalid entry; please correct "
                                       "by providing a positive integer:\n"))
            except:
                n_clusters = -1

        if method == 3:
            clustering_parameters = HIERARCHICAL_parameters('hierarchical',
                                                            n_clusters)
        else:
            clustering_parameters = KMEANS_parameters('k-means', n_clusters)

    answer = str(raw_input("\nECLAIR: the total number of consensus clusters defaults to the highest number of clusters encountered in each of the {:n} independent runs of subsamplings and clusterings. Do you want to provide a value instead? [Y/n]\n".format(N_runs)))
    if answer in {'y', 'Y', 'yes', 'Yes', 'YES', 'yep', 'Yep', 'YEP', 'Yoh, man!', 'si', 'Si', 'SI', 'oui', 'Oui', 'OUI', 'da','Da', 'DA'}:
        try:
            N_cc = int(input("\nECLAIR: how many clusters should make up "
                             "your consensus clustering?\n"))
        except:
            N_cc = 0
        while N_cc <= 0:
            try:
                N_cc = int(input("\nECLAIR: you should enter a positive integer. "
                                 "Please do so now:\n"))
            except:
                N_cc = 0

    elif answer in {'n', 'N', 'no', 'No', 'NO', 'nope', 'Nope', 'non', 'Non', 'NON', 'nada', 'Nada', 'NADA'}:
        N_cc = None

    else:
        N_cc = None
        print("\nECLAIR: let's assume this means the overall number of "
              "consensus clusters will be determined by default.\n")
        
    cc_parameters = CC_parameters(N_runs, sampling_fraction, N_cc)

    print('\n')

    time_now = datetime.datetime.today()
    format = "%Y-%m-%d %H:%M:%S"
    time_now = str(time_now.strftime(format))
    print('\nECLAIR\t INFO\t {}: ready to proceed!\n'.format(time_now))

    print('\n')

    return data_info, clustering_parameters, cc_parameters


def parse_options():
    """Specify the command line options to parse.
    
    Returns
    -------
    opts : optparse.Values instance
        Contains the option values in its 'dict' member variable.
    
    args[0] : string or file-handler
        The name of the file storing the data-set submitted
        for Affinity Propagation clustering.
    """

    parser = optparse.OptionParser(usage = "Usage: %prog [options] file_name\n\n"
                                   "file_name denotes the path for accessing the data "
                                   "submitted for ECLAIR analysis")

    parser.add_option('-s', '--skiprows', dest = 'skip_rows', type = 'int', 
                      help = ("The number of rows in the data file that should "
                      "be skipped as part of this file's header "
                      "[default: %default]"))

    parser.add_option('-i', '--cell_IDs_column', dest = 'cell_IDs_column', 
                      type = 'int', help = ("The column of the data file holding "
                      "information on the names, tags or IDs or the samples or "
                      "cells collected therein [default: %default]"))

    parser.add_option('-t', '--time_info_column', dest = 'time_info_column', 
                      type = 'int', help = ("The column recording the temporal "
                      "ordering or a time series of measurements for the dataset "
                      "to be processed [default: %default]"))

    parser.add_option('-e', '--excluded_columns', dest = 'extra_excluded_column', 
                      type = 'list', help = ("Specifies which features to remove, "
                      "in a kind of direct feature extraction; for some dataset "
                      "the user might want to consider more sophisticated feature "
                      "selection or feature extraction techniques, such as "
                      "Principal Component Analysis "
                      "or Sequential Backward Selection [default: %default]"))

    parser.add_option('-N', '--N_samples', dest = 'expected_N_samples', 
                      type = 'int', help = ("An estimate of the total number of "
                      "samples in your dataset; an accurate estimate helps "
                      "the Hierarchical Data Structure in allocating and chunking "
                      "some of the arrays involved in the ECLAIR machine learning "
                      "algorithm; an off-the-mark estimate however won't affect the "
                      "computations and output [default: %default]"))

    parser.add_option('-r', '--N_runs', dest = 'N_runs', type = 'int',
                      help = ("The size of the ensemble of partitions "
                      "to be aggregated and subjected to consensus clustering; "
                      "by extension, also the size of the ensemble of "
                      "minimum spanning trees subsequently pooled into "
                      "an ECLAIR graph/tree [default: %default]"))

    parser.add_option('-f', '--fraction', dest = 'sampling_fraction', 
                      type = 'float', help = ("The number of samples, expressed "
                      "as a fraction of the size of the whole dataset, to "
                      "randomly select at each run of density-based downsampling "
                      "and of clustering [default: %default]"))

    parser.add_option('-c', '--N_cc', dest = 'N_cc', type = 'int', 
                      help = ("The number of consensus clusters into which to "
                      "partition your dataset; this number doesn't have to agree "
                       "with the value of 'k' selected for k-means clustering, "
                       "for instance; in fact, consensus clustering can be applied "
                       "to an ensemble where samples have been partitioned into "
                       "different overall number of clusters [default: %default]"))

    parser.add_option('-m', '--method', dest = 'method', type = 'str', 
                      help = ("The clustering algorithm into which fraction of your "
                      "dataset are partitioned at each iteration of downsampling "
                      "and clustering; ECLAIR currently supports affinity "
                      "propagation, k-means and hierarchical clustering, "
                      "along with DBSCAN (Density-Based Spatial Clustering of "
                      "Applications with Noise) [default: %default]"))
                      
    parser.add_option('--convergence_iter', dest = 'convergence_iter', type = int,
                      help = ("If affinity propagation is the clustering "
                      "method of choice, specify the number of consecutive "
                      "iterations without change in the number of clusters that "
                      "signals convergence [default: %default]"))
    
    parser.add_option('--max_iter', dest = 'max_iter', type = int,
                      help = ("If affinity propagation has been chosen for each "
                      "round of subsampling and clustering, specify the maximum "
                      "number of message passing iterations undergone before "
                      "returning, having reached convergence or not "
                      "[default: %default]"))
                      
    parser.add_options('--minPts', dest = 'minPts', type = int,
                       help = ("If DBSCAN has been selected for clustering, "
                       "the number of points within an epsilon-radius "
                       "hypershpere for the said region to qualify as dense "
                       "[default: %default]"))
                       
    parser.add_options('--eps', dest = 'eps', type = float,
                       help = ("If DBSCAN has been selected for clustering, "
                       "the maximum distance separating two data-points "
                       "for those data-points to be considered as part of the "
                       "same neighborhood [default: %default]"))
                       
    parser.add_options('--quantile', dest = 'quantile', type = int,
                       help = ("If DBSCAN has been selected for clustering "
                       "and an 'eps' radius has not been provided by the user, "
                       "it will be determined as the 'quantile' of a distribution "
                       "of 'minPts'-nearest distances to each sample "
                       "[default: %default]"))
                       
    parser.add_options('--metric', dest = 'metric', type = 'str',
                       help = ("If DBSCAN has been selected for clustering, "
                       "the metric to use for computing the pairwise distances "
                       "between samples; if metric is a string or callable, "
                       "it must be compatible with "
                       "'metrics.pairwise.pairwise_distances' "
                       "[default: %default]"))
                       
    parser.add_options('-k', dest = 'k', type = int, help = ("The number of "
                       "clusters for 'k'-means clustering or agglomerative "
                       "clustering if selected as the method of choice for "
                       "each round of subsampling and clustering in ECLAIR "
                       "[default: %default]"))
    
    parser.set_defaults(skip_rows = 1, cell_IDs_column = 0, time_info_column = -1,
                        extra_excluded_columns = None, expected_N_samples = 100, 
                        N_runs = 50, sampling_fraction = '0.5', method = 'k-means',
                        convergence_iter = 15, max_iter = 150, minPts = 3, 
                        eps = None, quantile = 50, metric = 'minkowski', k = 50)

    opts, args = parser.parse_args()
    
    if len(args) == 0:
        parser.error('A data file must be specified')
        
    if opts.skip_rows < 0:
        parser.error("The number specifiying how many lines of header are in your file "
                     "must be non-negative")
                     
    if opts.expected_N_samples <= 0:
        parser.error("Please provide a non-negative estimate of the number of samples "
                     "in your dataset.")
                     
    if opts.N_runs < 1:
        parser.error("To take advantage of consensus clustering, please subject your "
                     "dataset to at least two rounds of downsampling and clustering.")
                     
    if not (0 < sampling_fraction < 1):
        parser.error("The sampling fraction parameter must be in the range (0; 1).")
        
    if opts.N_cc < 2:
        parser.error("The desired number of consensus clusters must be at least 2.")
        
    return opts, args[0] 


def main():

    locale.setlocale(locale.LC_ALL, "")
        
    if len(sys.argv) == 1:
    # Calling the user interface:
        data_info, clustering_parameters, cc_parameters = user_interface()
    else:
        opts, args = parse_options()
    
        data_info = Data_info(args, opts.expected_N_samples, opts.skip_rows,
                              opts.cell_IDs_column, opts.extra_excluded_columns,
                              opts.time_info_column)

        cc_parameters = CC_parameters(opts.N_runs, opts.sampling_fraction, opts.N_cc)
    
        if opts.method in {'KMEANS', 'k-means', 'kmeans'}:
            clustering_parameters = KMEANS_parameters('k-means', opts.k)
        elif opts.method in {'DBSCAN', 'dbscan'}:
            clustering_parameters = DBSCAN_parameters('DBSCAN', opts.minPts, opts.eps,
                                                      opts.quantile, opts.metric)
        elif opts.method in {'Affinity Propagation', 'Affinity propagation', 
                             'Affinity-Propagation', 'Affinity-propagation', 
                             'affinity propagation', 'affinity-propagation'}:
            clustering_parameters = AP_parameters('affinity_propagation', opts.max_iter,
                                                  opts.convergence_iter)
        elif opts.method in {'Agglomerative', 'Hierarchical', 'hierarchical',
                             'agglomerative'}:
            clustering_parameters = HIERARCHICAL_parameters('hierarchical', opts.k)
        else:
            print("\nERROR: ECLAIR: Build_instance: invalid choice of clustering "
                  "method\n")
            sys.exit(1)
    
    try:
        os.makedirs('./ECLAIR_instance')
    except OSError:
        if not os.path.isdir('./ECLAIR_instance'):
            print("\nERROR: ECLAIR: Build_instance: failed to create a directory "
                  "where to store all the information pertaining to the "
                  "ECLAIR statistical learning on the dataset provided at input.\n")
            raise

    # Ready to start generating an instance of ECLAIR!
    _ = ECLAIR_processing('./ECLAIR_instance/store.h5', 
                          data_info, clustering_parameters, 
                          cc_parameters, './ECLAIR_instance')
                      

if __name__ == '__main__':
    main()
    
    
