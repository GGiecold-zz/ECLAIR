#!/usr/bin/env python


# ECLAIR/setup.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""Setup script for ECLAIR, a package for the robust and scalable 
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


from codecs import open
from os import path
from sys import exit, version
from setuptools import setup
from setuptools.command.install import install
import subprocess


here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding = 'utf-8') as f:
    long_description = f.read()
    

setup(name = 'ECLAIR',
      version = '1.18',
      
      description = "Robust inference of cell lineages from gene expression data " 
                    "via consensus clustering and the aggregation of ensembles "
                    "of minimum spanning trees.",
      long_description = long_description,
                    
      url = 'https://github.com/GGiecold/ECLAIR',
      download_url = 'https://github.com/GGiecold/ECLAIR',
      
      author = 'Gregory Giecold',
      author_email = 'g.giecold@gmail.com',
      maintainer = 'Gregory Giecold',
      maintainer_email = 'ggiecold@jimmy.harvard.edu',
      
      license = 'MIT License',
      
      platforms = ('Any',),
      install_requires = ['Cluster_Ensembles>=1.16', 'Concurrent_AP>=1.3', 
                          'DBSCAN_multiplex>=1.5', 'Density_Sampling>=1.1',
                          'matplotlib>=1.4.3', 'munkres', 'numpy>=1.9.0',
                          'psutil', 'python-igraph', 'scipy>=0.16', 'sklearn', 
                          'setuptools', 'tables'],
                          
      classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',          
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Mathematics', ],
                   
      packages = ['ECLAIR'],
      package_dir = {'ECLAIR': 'src/ECLAIR'},
      
      include_package_data = True,
      package_data = {
          'ECLAIR': 
              ['data/Guoji_data/qPCR.txt'
               'data/SPADE_data/nbt-SD2.csv'
               'data/SPADE_data/nbt-SD2.fcs'
               'data/SPADE_data/nbt-SD2-Transformed.csv'    
              ],
      },
      
      keywords = "aggregation bio-informatics clustering computational-biology "
                 "consensus consensus-clustering cytometry data-mining "
                 "ensemble ensemble-clustering gene gene-expression "
                 "graph graph-matching machine-learning matching "
                 "pattern-recognition qPCR tree tree-matching "
                 "unsupervised-learning",
                 
      entry_points = {
          'console_scripts': 
              ['ECLAIR_make = ECLAIR.Build_instance.__main__',
               'ECLAIR_performance = ECLAIR.Statistical_performance.__main__',
              ],
      }                    
)

