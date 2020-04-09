# ExtraCluster: Addons for clustering

Some clustering algorithms that haven't implemented in scikit-learn, and are proposed especifically for face clustering.
**Extra point**. I have combined the Approximate-Rank-Order algorithm and the Chinese-Whispers algorithm as a new algorithm which i believe performs better than its two parents: **_ROCWClustering_**. This algorithm generates a distnce graph using rank-order distances, and then the data points have been clustered by feeding this graph to ChineseWhispers algorithm.


# Requirements
```
1. numpy
2. scikit-learn
```

# How to use?
It is just as simple as instantiating a clustering algorithm, and calling its `fit_predict` method on provided data, just like the scikit-learn estimators.

See [this notebook](https://github.com/iamsoroush/clustering/blob/master/examples.ipynb) for some simple usage examples.
