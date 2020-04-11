# ExtraCluster: Addons for clustering

Some clustering algorithms that haven't implemented in scikit-learn, and are proposed especifically for face clustering.

**Extra point**. I have combined the Approximate-Rank-Order algorithm and the Chinese-Whispers algorithm as a new algorithm which i believe performs better than its two parents: **_ROCWClustering_**. This algorithm generates a distnce graph using rank-order distances, and then the data points have been clustered by feeding this graph to ChineseWhispers algorithm.


Here are the source papers:
1. [Chinese Whispers](https://pdfs.semanticscholar.org/c64b/9ed6a42b93a24316c7d1d6b\3fddbd96dbaf5.pdf?_ga=2.16343989.248353595.1538147473-1437352660.1538147473)
2. [Rank-Order](https://ieeexplore.ieee.org/document/5995680)
3. [Approximate Rank-Order](https://arxiv.org/abs/1604.00989)


# Requirements
```
1. numpy
2. scikit-learn
```

# How to use?
It is just as simple as instantiating a clustering algorithm, and calling its `fit_predict` method on provided data, just like the scikit-learn estimators.

See [this notebook](https://github.com/iamsoroush/clustering/blob/master/examples.ipynb) for some simple usage examples.


# Results
Here's the performances on some dummy datasets. Note that the data has a low-dimensionality, whereas these algorithms especially perform well for higher dimensional data spaces. Furthermore, note that no parameter optimisation have been used.

![alt text](https://github.com/iamsoroush/clustering/blob/master/index.png "Results")



