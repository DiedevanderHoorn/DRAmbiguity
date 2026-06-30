# DRAmbiguity
## When One Point Is Not Enough: Addressing Ambiguous Instances in Dimensionality Reduction by Splitting

Code for identifying and resolving ambiguous instances in dimensionality reduction via graph-based splitting.

This repository contains the code accompanying our [paper](https://arxiv.org/pdf/2605.23540).

### Abstract
Dimensionality Reduction (DR) methods are widely used to visualize high-dimensional data. One key task in DR-based analysis is discovering neighborhoods, which relies on analyzing the fine-grained local structure of a projection. However, DR is an inherently lossy process; no technique can perfectly preserve the high-dimensional relationships, and projections therefore contain visual artifacts. In this paper, we highlight a typically overlooked source of visual artifacts: ambiguous instances. These are instances that are highly similar to multiple mutually dissimilar neighborhoods in the high-dimensional space. Standard DR methods cannot faithfully project such instances, since each data instance is mapped to a single point in the visual space. As a result, such an instance is placed in only one of its neighborhoods (or in none at all), so only part of its neighborhood structure is represented. We call this distortion partial neighborhood embedding. In this paper, we introduce a graph-based approach that identifies ambiguous instances and replicates them as multiple points in the projection, placing each copy within its respective neighborhood. We use UMAP for our results, but our approach also generalizes to other local graph-based DR techniques, and we show that our approach reveals previously hidden neighborhood memberships in projections and reduces partial neighborhood embedding across multiple examples, and is further supported by quantitative analyses.

## Structure
The `GraphDisambiguation` class in the `graphdisambiguation.py` file contains the main part of the implementation. 
Each figure in the paper corresponds to a script named `fig_*.py`. The data folder contains all datasets used (sometimes preprocessed).

## Reproducing paper results
To reproduce the results from the paper, please do the following:

1. Clone this repository
2. Install Julia (version 1.12.5) (required for sparsification implementation). Can be downloaded [here](https://julialang.org/downloads/).
3. Install dependencies: ```pip install -r requirements-lock.txt```
4. Run any of the `fig_*.py` scripts to generate the figures (or components that were used to create the full figures)

Note that on different systems there may be some slight variation of the results. The results in the paper were generated using Python 3.11.13 on Windows.

## Dataset sources:
- **Case-based reasoning**: F. V. Paulovich, L. G. Nonato, R. Minghim, and H. Levkowitz. Least square projection: A fast high-precision multidimensional projection technique and its application to document mapping. IEEE Transactions on Visualization and Computer Graphics, 14(3):564–575, 2008. [doi](https://doi.org/10.1109/TVCG.2007.70443)
- **MNIST**: Y. LeCun, P. Haffner, L. Bottou, and Y. Bengio. Object recognition with gradient-based learning. In Shape, contour and grouping in computer vision, pp. 319–345. Springer, 1999. [doi](https://doi.org/10.1007/3-540-46805-6_19)
- **RNA-seq**: E. Becht, L. McInnes, J. Healy, C.-A. Dutertre, I. W. H. Kwok, L. G. Ng et al. Dimensionality reduction for visualizing single-cell data using UMAP. Nature Biotechnology, 37(1):38–44, 2019. [doi](https://doi.org/10.1038/nbt.4314)
- **Street View House Numbers**: Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, 2011.
