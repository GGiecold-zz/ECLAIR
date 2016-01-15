#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistical_performance/__init__.py;

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


from .Convergence_analysis import *
from .Robustness_analysis import *
from .Statistical_tests import *

def tree_edges(ref_tree, max_ref_dist, max_overall_dist):
    from .Class_tree_edges import tree_edges
    return tree_edges(ref_tree, max_ref_dist, max_overall_dist)
    
def gaussian_kde(data_hdf5_storage, bandwidth_method = None):
    from .Gaussian_KDE_HDF5 import gaussian_kde
    return gaussian_kde(data_hdf5_storage, bandwidth_method)


__all__ = ['Class_tree_edges', 'Convergence_analysis', 'Gaussian_KDE_HDF5',
           'Robustness_analysis', 'Statistical_tests']

