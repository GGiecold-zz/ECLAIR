# ECLAIR
Robust and scalable inference of cell lineages from gene expression data.

ECLAIR achieves a higher level of confidence in the estimated lineages through the use of approximation algorithms for consensus clustering and by combining the information from an ensemble of minimum spanning trees so as to come up with an improved, aggregated lineage tree. 

In addition, the present package features several customized algorithms for assessing the similarity between weighted graphs or unrooted trees and for estimating the reproducibility of each edge to a given tree.

How ECLAIR graphs and trees are generated
-----------------------------------------

ECLAIR stands for Ensemble Cell Lineage Analysis with Improved Robustness. It proceeds as follow:
* Choose among affinity propagation, hierarchical or k-means clustering, along with DBSCAN (cf. our ```DBSCAN_multiplex``` and ```Concurrent_AP``` packages for streamlined and scalable implementations of DBSCAN and affinity propagation clustering) for how to partition a subsample of your dataset.

* Such a subsample is obtained by density-based downsampling (as implemented in our ```Density_Sampling``` software posted on the Python Package Index), either by aiming for an overall number of datapoints to extract from the dataset or by specifiying a target percentile of the distribution of local densities around each datapoint.

* ECLAIR then goes about performing several rounds of downsampling and clustering on such subsamples, for as many iterations as specified by the user. After each run of clustering of one such subsample, the datapoints left over are upsampled by associating them to the closest centroid in high-dimensional feature space.

* For each such run, build a minimum spanning tree providing a path among the clusters. Such a minimum spanning tree is obtained from a matrix of L2 pairwise similarities between the centroids associated to each cluster. 

* The next step seeks a consensus clustering from this ensemble of partitions of the whole dataset. Three heuristic methods are considered for this purpose: CSPA, HGPA and MCLA, all of them based on graph or hypergraph partitioning (cf. the documentation of our ```Cluster_Ensembles``` package for more information).

* Once a consensus clustering has been reached, we build a graph from its clusters. The edge weights of this graph are built as the mean of the following distribution: for each of the 2-uple consisting of one datapoint from consensus cluster ```a``` and another datapoint from consensus cluster ```b```, scan over the ensemble of partitions and keep track of the distance separating those two samples for each run making up this ensemble. This distance is computed as the number of separating edges along a given tree from the ensemble of minimum spanning trees (only their topology matters at this point, even though those trees are obtained from a matrix of pairwise similarities in gene expression space and their edges could also be viewed as bearing weights extracted from this matrix).

* We then obtain a minimum spanning tree from this graph, for convenience of visualization as well as for later comparison with other methods purporting to provide good estimates of cell lineages.

Statistical performance of ECLAIR
---------------------------------

To compare two lineage trees, one has to take into account their edge connections but also the sample contents of their nodes, since the variation associated to subsampling results in different clusters of samples. Although there are many papers on graph matching and graph comparison, we are not aware of any previously published method that takes into account the node differences. We therefore developed new statistical tests suitable for comparing lineage trees. 

* First, we define a metric to compare the overall similarity between two lineage trees, ```T_1``` and ```T_2```. For each tree, we evaluate the path length between every pair of cells in the population, based on the edge connectivity. The correlation between the two sets of path length values is used as a metric to compare the overall similarity of ```T_1``` and ```T_2```. For a moderately large dataset of 500,000 samples, this would naively translate into more than 100 billion pairs of distances along ```T_1```and along ```T_2```. The details of the much more efficient algorithm we developped for that purpose is available from the docstrings of our package; the gist of this algorithm is to first build a contingency table recording the overlap in the number of samples between pairs of ```T_1``` nodes versus pairs of ```T_2``` nodes.

* Second, we define ```D_ij```as an edge-specific measures of statistical dispersion to evaluate the robustness of each edge within a given lineage tree , denoted ```T*```. Specifically, for each edge ```E_ij``` connecting a pair of clusters ```C_i*``` and ```C_j*```, we define the dispersion ```D_ij``` associated with ```E_ij``` as the standard deviation of the the distribution of path lengths ```L^a(x,y)```, where ```x``` and ```y``` are selected from ```C_i*```and ```C_j*```  respectively, and ```a``` is summed over the partitions and minimum spanning trees from the ensemble out of which ```T*``` was constructed in the first place. 

* The afore-mentioned measure of statistical dispersion is computed solely in terms of the partitions and trees making up an ensemble from which a consensus clustering and an ECLAIR tree are then extracted. We also compare this measure with another measure of statistical dispersion, obtained by independently generating 50 different ECLAIR trees. One such tree is singled out as a reference tree. For each edge of this reference tree, we keep track of how far spread out the pairs of cells comprising the two nodes of this reference edge are across the rest of the 49 ECLAIR tree. 

Our ECLAIR package features a module that computes such statistical measures and a few more tests on pairs of ECLAIR trees.

Installation
------------

ECLAIR is written in Python 2.7. Install ECLAIR by sending a request to the Python Package Index (PyPI) as follows:
* start a terminal;
* enter ```pip install ECLAIR```.

Any missing third-party dependency should be automatically resolved. Apart from the Python Standard Library, those include:
* ```Cluster_Ensembles``` (version 1.16 or later)
* ```Concurrent_AP``` (version 1.3 or later)
* ```DBSCAN_multiplex``` (version 1.5 or ulterior)
* ```Density_Sampling``` (1.1 or subsequent version)
* ```matplotlib``` (version 1.4.3 at least)
* ```munkres```
* ```numpy``` (1.9.0 or ulterior version)
* ```scipy```
* ```sklearn```
* ```setuptools```
* ```tables```

Please note that as part of the installation of this package, some code written in C that is part of the ```Cluster_Ensembles``` package will be automatically compiled, under the hood and according to the specifications of your machine. For this process to go seamlessly, you have however to ensure availability of CMake and GNU make on your operating system. ```Cluster_Ensembles``` also requires the 32-bit version of the GNU C library. Please refer to the ```Cluster_Ensembles``` documentation for more information on how to meet those few requirements depending on Linux distribution.

Usage
-----

To subject a dataset to an ECLAIR analysis:
* start a terminal;
* enter ```ECLAIR_make [options] file_name```, where ```file_name``` denotes the path to the data about to be processed.
It is generally recommended to leave the ```options``` field empty, which will prompt an interface guiding the user through the choice of parameters for the ECLAIR analysis at hand.

To launch a full-fledged statistical performance analysis of ECLAIR and see how it consistenly performs better than SPADE, a popular method for estimating cell lineages, proceed as follows:
* at the Shell command-line interface or graphical user interface, type in ```ECLAIR_performance```.

In the current version, the statisitcal performance of ECLAIR is only evaluated for a moderately large flow cytometry dataset of half-a-million samples and 8 features, as well as on a qPCR dataset of mouse bone marrow samples. It shouldn't be difficult for anyone competent in Python to peer through the source code of ECLAIR and bring about a few of the changes required to submit his/her own dataset to a similar statisitcal analysis (those changes mostly pertain to domain-specific knowledge and to the format of your dataset).

Several "experiments" will thereby be performed, including the comparisons of pairs of ECLAIR graphs or trees and pairs of SPADE trees generated on the same dataset. The comparison of ECLAIR instances and of SPADE instances generated on non-overlapping datasets and evaluated on a separate test set calls for detailed explanations. 

To that purpose, we are splitting a dataset into three equally-sized, non-overlapping parts, ```S1```, ```S2``` and ```S3```. We train an ECLAIR tree (```Ecl_1```) and a SPADE tree on ```S1``` (```Spd_1```). We then train another ECLAIR tree (```Ecl_2```) and yet another SPADE tree (```Spd_2```) on the set ```S2```.

The training procedure for ```Ecl_1``` involves 50 runs of downsampling and clustering of the samples within ```S1```. The downsampling ratio is set at 50%. Therefore, ```Ecl_1``` is an aggregation of 50 trees, all generated from ```S1``` alone.

In order to compare ```Ecl_1``` with ```Ecl_2```, the cells in ```S3``` are mapped to the clusters/nodes in ```Ecl_1``` and in ```Ecl_2``` to which they are nearest in the high-dimensional gene expression space.

Idem when it comes to comparing ```Spd_1``` and ```Spd_2```.

The procedure outlined above is repeated 10 times. We end up with two lists of 30 correlation coefficients telling us about the similarity of as many pairs of ECLAIR or SPADE trees. Indeed, while things have been exposed as involving only the evaluation of ```Ecl_1``` and ```Ecl_2``` on ```S3``` using as a test set, one can also generate an ECLAIR tree using S3 as a training set. This allows the additional comparisons of ```Ecl_1``` with ```Ecl_3``` and of ```Ecl_2``` with ```Ecl_3```.

It also bears pointing out we are using the same test set (```S3```) for assessing the similarity of pairs of ECLAIR trees (```Ecl_1``` vs. ```Ecl_2```) as for evaluating the similitude of pairs of SPADE trees (```Spd_1``` vs. ```Spd_2```).

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
