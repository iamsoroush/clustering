# -*- coding: utf-8 -*-
"""Randk-Order-Based clustering algorithm.

This file contains classes that are implementations of two clustering algorithms
 proposed for face clustering problem.

I combine Approximate Rank-Order algorithm and Chinese Whispers algorithm
 in ROCWClustering class.
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

from helpers import Cluster, ClusterContainer, ChineseWhispersClustering, ROGraph


class BaseClustering:
    def __init__(self):
        pass

    def fit_predict(self, X):
        pass

    def score(self, X, y_true):
        y_pred = self.fit_predict(X)
        pairwise_precision = self._calc_pw_precision(y_pred, y_true)
        pairwise_recall = self._calc_pw_recall(y_pred, y_true)
        pairwise_f_measure = 2 * (pairwise_precision * pairwise_recall)\
            / (pairwise_precision + pairwise_recall)
        return pairwise_f_measure, pairwise_precision, pairwise_recall

    @staticmethod
    def _calc_pw_precision(y_pred, y_true):
        unique_clusters = np.unique(y_pred)
        n_pairs = 0
        n_same_class_pairs = 0
        for cluster in unique_clusters:
            sample_indices = np.where(y_pred == cluster)[0]
            combs = np.array(list(itertools.combinations(sample_indices, 2)))
            combs_classes = y_true[combs]
            same_class_pairs = np.where(combs_classes[:, 0] == combs_classes[:, 1])[0]
            n_pairs += len(combs)
            n_same_class_pairs += len(same_class_pairs)
        pw_precision = n_same_class_pairs / n_pairs
        return pw_precision

    @staticmethod
    def _calc_pw_recall(y_pred, y_true):
        unique_classes = np.unique(y_true)
        n_pairs = 0
        n_same_cluster_pairs = 0
        for clss in unique_classes:
            sample_indices = np.where(y_true == clss)[0]
            combs = np.array(list(itertools.combinations(sample_indices, 2)))
            combs_clusters = y_pred[combs]
            same_cluster_pairs = np.where(combs_clusters[:, 0] == combs_clusters[:, 1])[0]
            n_pairs += len(combs)
            n_same_cluster_pairs += len(same_cluster_pairs)
        pw_recall = n_same_cluster_pairs / n_pairs
        return pw_recall


class ROCWClustering(BaseClustering):
    """Approximated rank-order clustering implemented using Chinese Whispers algorithm.

    Using rank-order distances generate a graph, and feed this graph to ChineseWhispers
     algorithm for clustering.

    """

    def __init__(self, k=20, metric='euclidean', n_iteration=5, algorithm='ball_tree'):
        self.k = k
        self.metric = metric
        self.n_iteration = n_iteration
        self.knn_algorithm = algorithm

    def fit_predict(self, X):
        graph = ROGraph(self.k, self.metric, algorithm=self.knn_algorithm)
        adjacency_mat = graph.generate_graph(X)
        clusterer = ChineseWhispersClustering(self.n_iteration)
        labels = clusterer.fit_predict(adjacency_mat)
        return labels


class RankOrderClustering(BaseClustering):
    """Class for rank-order clustering algorithm.

    Source paper:  DOI: 10.1109/CVPR.2011.5995680

    Create an instance, and call the 'fit_predict' method.
    """

    def __init__(self, metric='euclidean', threshold=10, k=20):
        self.ranking_metric = metric
        self.threshold = threshold
        self.k = k

    def fit_predict(self, X):
        """Finds clusters and returns founded labels.

        Parameters
        ----------
        X: :obj: np.ndarray
            An array of data samples.

        returns: An array containing predicted label for each data point.
        """

        n_samples = X.shape[0]

        # Generate order list for every data point
        nbrs = NearestNeighbors(n_neighbors=n_samples, algorithm='ball_tree',
                                metric=self.ranking_metric).fit(X)
        ordered_absolute_distances, sample_level_order_lists = nbrs.kneighbors(X)
        absolute_distances = pairwise_distances(X, metric=self.ranking_metric)

        # Initialize clusters, each data point will be a separate cluster
        cluster_list = [Cluster(label=i, members=[i]) for i in range(n_samples)]
        clusters = ClusterContainer(init_clusters=cluster_list,
                                    absolute_distances=absolute_distances,
                                    sample_level_order_lists=sample_level_order_lists)

        # Do rank-order clustering
        while True:

            # Create every possible pair of clusters, based on indices in "clusters" object
            cluster_labels = [i for i in range(len(clusters))]
            pairs = itertools.combinations(cluster_labels, 2)

            # Candidate pairs will be stored in a list of tuples, each tuple containing
            # two cluster index
            candidate_pairs = list()

            # For each possible pair of clusters, calculate distances and check
            # the merging conditions, and add to candidate pairs if the conditions
            # satisfied.
            for cluster_ind_1, cluster_ind_2 in pairs:
                dist1 = clusters.cluster_level_absolut_distances[cluster_ind_1]
                dist2 = clusters.cluster_level_absolut_distances[cluster_ind_2]

                d_r, d_n = self.calc_dr_dn(clusters.clusters[cluster_ind_1],
                                           clusters.clusters[cluster_ind_2],
                                           dist1, dist2,
                                           ordered_absolute_distances,
                                           self.k)
                if (d_r < self.threshold) and (d_n < 1):
                    candidate_pairs.append((cluster_ind_1, cluster_ind_2))
            if candidate_pairs:
                clusters_to_merge = self.find_clusters_to_merge(candidate_pairs)

                # Merge clusters and update cluster-level absolute distances
                clusters.merge_clusters(clusters_to_merge)
            else:
                break

        labels = np.zeros(X.shape[0])
        for cluster in clusters:
            indices = cluster.members
            labels[indices] = cluster.label
        return labels

    def find_clusters_to_merge(self, candidate_pairs):
        clusters_to_merge = []
        pairs = candidate_pairs.copy()
        while pairs:
            pair = pairs.pop()
            pairs_copy = pairs.copy()
            new_cluster = list(pair)
            i = 0
            while True:
                current_cluster = new_cluster[i]
                for p in pairs_copy:
                    if current_cluster in p:
                        if current_cluster == p[0]:
                            new_cluster.append(p[1])
                        else:
                            new_cluster.append(p[0])
                        pairs.remove(p)
                pairs_copy = pairs.copy()
                i += 1
                if i >= len(new_cluster):
                    break
            clusters_to_merge.append(list(set(new_cluster)))
        return clusters_to_merge

    def calc_dr_dn(self, c1, c2, dist1, dist2, ordered_absolute_distances, k):
        """Calculate two distance metrics in cluster-level.

        Parameters
        ----------
        c1 : :obj: Cluster
        c2 : :obj: Cluster
        ol1 : :obj: np.ndarray
        ol2 : :obj: np.ndarray
        absolute_distances : :obj: np.ndarray . array of arrays.
        k : int
        """

        ol1 = np.argsort(dist1)
        ol2 = np.argsort(dist2)
        o1, d1 = self.calc_asym_dist(c1.label, c2.label, ol1, ol2)
        o2, d2 = self.calc_asym_dist(c2.label, c1.label, ol2, ol1)
        d_r = (d1 + d2) / min(o1, o2)

        phi = self.compute_phi(c1, c2, ordered_absolute_distances, k)
        d_n = dist1[c2.label] / phi
        return d_r, d_n

    @staticmethod
    def compute_phi(cluster1, cluster2, ordered_absolute_distances, k):
        data_indices = list()
        data_indices.extend(cluster1.members)
        data_indices.extend(cluster2.members)
        sum_1 = 0
        for sample_ind in data_indices:
            sum_1 += ordered_absolute_distances[sample_ind, 0: k].sum() / k
        return sum_1 / (len(cluster1) + len(cluster2))

    @staticmethod
    def calc_asym_dist(ind1, ind2, ol1, ol2):
        b_in_a = np.where(ol1 == ind2)[0][0]
        d = 0
        for i in range(b_in_a):
            f = ol1[i]
            o = np.where(ol2 == f)[0][0]
            d += o
        return b_in_a, d


class ApproximateRankOrderClustering(BaseClustering):
    """Approximate rank-order clustering implementation.

    Source paper: https://arxiv.org/abs/1604.00989
    """

    def __init__(self, k=50, threshold=0.1, metric='euclidean'):
        """Init an instance.

        Parameters
        ----------
        k : Number of nearest neoghbors

        threshold : A parameter for merging clusters. All pairs of clusters
         with distances below this threshold will merge transitively.

        absolute_distance_metric : Distance metric to pass to NearestNeighbors estimator .
        """

        self.k = k
        self.threshold = threshold
        self.absolute_distance_metric = metric

    def fit_predict(self, X):
        """Finds clusters within given data and returns predicted labels.

        Parameters
        ----------
        X: array of np.ndarray objects. i.e. features extracted for given faces using
            facenet architecture.

        returns: labels for input samples, indicating corresponding clusters.
        """

        n_samples = X.shape[0]
        ordered_distances, order_lists = self.get_knns(X)
        pairwise_distances = self.compute_pairwise_distances(ordered_distances,
                                                             order_lists)
        clusters = self.merge(pairwise_distances)
        labels = np.zeros(n_samples)
        for i, c in enumerate(clusters):
            labels[c] = i
        return labels

    def get_knns(self, X):
        """Generates order lists and absolute distances of k-nearest-neighbors
            for each data point.
        """

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree',
                                metric=self.absolute_distance_metric).fit(X)
        ordered_absolute_distances, order_lists = nbrs.kneighbors(X)
        return ordered_absolute_distances, order_lists

    def compute_pairwise_distances(self, ordered_distances, order_lists):
        """Returns a matrix of shape (n_samples, n_samples) of pw distances.
        Elements that do not share any neighbors will filled by 'np.Inf'.
        """

        distance_measures = self._compute_distance_measures(order_lists, self.k)
        pairwise_distances = self._compute_pw_dist(distance_measures, order_lists)
        self.dist_measures = distance_measures
        self.pw_dist = pairwise_distances
        return pairwise_distances

    @staticmethod
    def _compute_distance_measures(order_lists, k):
        """Returns a matrix of shape (n_samples, n_samples) with each element
            representing i-to-j distance.
        """

        n_samples = len(order_lists)
        distance_measures = np.ones((n_samples, n_samples)) * np.Inf
        np.fill_diagonal(distance_measures, 0)
        for ind1, order_list in enumerate(order_lists):
            for ind2, neighbor in enumerate(order_list):
                dm = 0
                for sample in order_list[:ind2]:
                    if sample in order_lists[ind2]:
                        continue
                    else:
                        dm += 1
                distance_measures[ind1, ind2] = dm
        return distance_measures

    @staticmethod
    def _compute_pw_dist(dist_measures, order_lists):
        n_samples = len(order_lists)
        pw_dist = np.ones((n_samples, n_samples)) * np.Inf
        np.fill_diagonal(pw_dist, 0)
        combs = itertools.combinations([i for i in range(n_samples)], 2)
        for i, j in combs:
            if np.any(order_lists[i] == j) or np.any(order_lists[j] == i):
                o_i = np.where(order_lists[i] == j)[0]
                o_j = np.where(order_lists[j] == i)[0]
                denom = 0
                if o_i.size:
                    if o_j.size:
                        denom = min(o_i[0], o_j[0])
                    else:
                        denom = o_i[0]
                else:
                    denom = o_j[0]
                d_i = dist_measures[i, j]
                d_j = dist_measures[j, i]
                dist = (d_i + d_j) / denom
                pw_dist[i, j] = dist
                pw_dist[j, i] = dist
        return pw_dist

    def merge(self, pw_distances):
        """Transitively merge all pairs of samples with pw distances below self.threshold ."""

        indices = [i for i in range(pw_distances.shape[0])]
        clusters = []
        while indices:
            ind = indices.pop()
            indices_copy = indices.copy()
            cluster = [ind]
            adjacents = np.where(pw_distances[ind] < self.threshold)[0]
            if np.any(adjacents):
                for i in indices_copy:
                    if i in adjacents:
                        indices.remove(i)
                        cluster.append(i)
            clusters.append(cluster)
        return clusters
