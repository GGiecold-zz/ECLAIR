# ECLAIR
Robust and scalable inference of cell lineages from gene expression data.

ECLAIR achieves a higher level of confidence in the estimated lineages through the use of approximation algorithms for consensus clustering and by combining the information from an ensemble of minimum spanning trees so as to come up with an improved, aggregated lineage tree. 

In addition, the present package features several customized algorithms for assessing the similarity between weighted graphs or unrooted trees and for estimating the reproducibility of each edge to a given tree.

Overview of ECLAIR
------------------

ECLAIR stands for Ensemble Cell Lineage Analysis with Improved Robustness. It proceeds as follow:
* Choose among affinity propagation, hierarchical or k-means clustering, along with DBSCAN (cf. our ```DBSCAN_multiplex``` and ```Concurrent_AP``` packages for streamlined and scalable implementations of DBSCAN and affinity propagation clustering) for how to partition a subsample of your dataset.
* Such a subsample is obtained by density-based downsampling (as implemented in our ```Density_Sampling``` software posted on the Python Package Index), either by aiming for an overall number of datapoints to extract from the dataset or by specifiying a target percentile of the distribution of local densities around each datapoint.
* ECLAIR then goes about performing several rounds of downsampling and clustering on such subsamples, for as many iterations as specified by the user. After each run of clustering of one such subsample, the datapoints left over are upsampled by associating them to the closest centroid in high-dimensional feature space.
* For each such run, build a minimum spanning tree providing a path among the clusters. Such a minimum spanninh tree is obtained from a matrix of L2 pairwise similarities between the centroids associated to each cluster. 
* The next step seeks a consensus clustering from this ensemble of partitions of the whole dataset. Three heuristic methods are considered for this purpose: CSPA, HGPA and MCLA, all of them based on graph or hypergraph partitioning (cf. the documentation of our ```Cluster_Ensembles``` package for more information).
* Once a consensus clustering has been reached, we build a graph from its clusters. The edge weights of this graph are built as the mean of the following distribution: for each of the 2-uple consisting of one datapoint from consensus cluster 'a' and another datapoint from consensus cluster 'b', scan over the ensemble of partitions and keep track of the distance separating those two samples for each run making up this ensemble. This distance is computed as the number of separating edges along a given tree from the ensemble of minimum spanning trees (only their topology matters at this point, even though those trees are obtained from a matrix of pairwise similarities in gene expression space and their edges could also be viewed as bearing weights extracted from this matrix).
* We then obtain a minimum spanning tree from this graph, for convenience of visualization as well as for later comparison with other methods purporting to provide good estimates of cell lineages.

References
----------
* Giecold, G., Marco, E., Trippa, L. and Yuan, G.-C.,
"Robust Inference of Cell Lineages". 
Submitted for publication
* Strehl, A. and Ghosh, J., "Cluster Ensembles - A Knowledge Reuse Framework
for Combining Multiple Partitions".
In: Journal of Machine Learning Research, 3, pp. 583-617. 2002
* Conte, D., Foggia, P., Sansone, C. and Vento, M., "Thirty Years of Graph Matching in Pattern Recognition".
In: International Journal of Pattern Recognition and Artificial Intelligence, 18, 3, pp. 265-298. 2004

IMPORTANT NOTICE
----------------

More details, along with installation instructions to appear soon!
