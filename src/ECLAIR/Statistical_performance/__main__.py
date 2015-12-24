#!/usr/bin/env python


# ECLAIR/src/ECLAIR/Statistical_performance/__main__.py;

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
Giecold, G., Marco, E., Trippa, L. and Yuan, G.-C.,
"Robust Inference of Cell Lineages".
Submitted for publication

Strehl, A. and Ghosh, J., "Cluster Ensembles - A Knowledge Reuse Framework
for Combining Multiple Partitions".
In: Journal of Machine Learning Research, 3, pp. 583-617. 2002

Conte, D., Foggia, P., Sansone, C. and Vento, M., 
"Thirty Years of Graph Matching in Pattern Recognition". 
In: International Journal of Pattern Recognition and Artificial Intelligence, 
18, 3, pp. 265-298. 2004
"""


from . import Robustness_analysis

from itertools import permutations

# i.   Split a dataset in three non-overlapping, equally-sized parts, S1, S2, S3;
# ii.  Generate and ECLAIR tree/graph or a SPADE tree on S1, then another on S2;
# iii. Compare the afore-mentioned pairs of trees on S3, viewed as a test set.
# iv.  Repeat sets i. to iii. by interverting the roles of S1, S2 and S3
#      as training and test sets.
# v.   Repeat steps i. to iv. up to 10 times so as to generate a series of 
#      coefficients ascertaining the similarity of trees or graphs.
for data_flags in sorted(permutations([True, False, False]))[::-2]:
    method = 'hierarchical' if data_flags[-1] is True else 'k-means'
    experiment_1(3, data_flags, method)

# Pairwise comparisons of ECLAIR trees/graphs generated on the same dataset
experiment_2('./ECLAIR_performance/nbt-SD2-Transformed.csv', 
             k = 50, sampling_fraction = 0.5, N_runs = 100)
       
# Pairwise comparisons of SPADE trees generated on the same dataset             
experiment_3()

