# -*- coding: utf-8 -*-
"""Helper classes for algorithms implemented in src.py
"""
# Author: Soroush Moazed <soroush.moazed@gmail.com>

from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors


class ClusterContainer:
    """A container for clusters.

    Takes a list of "Cluster" objects and implements some methods on clusters
    """

    def __init__(self, init_clusters, absolute_distances, sample_level_order_lists):
        """Initializer.

        Parameters
        ----------
        init_clusters: A list of "Cluster" objects

        absolute_distances: :list of arrays:
        """

        self.clusters = init_clusters
        self.absolute_distances = absolute_distances
        self.sample_level_order_lists = sample_level_order_lists
        self.cluster_level_absolut_distances = absolute_distances

    def merge_clusters(self, merge_list):
        new_clusters = []
        clusters_to_remove = []
        for cluster_indices in merge_list:
            clusters = [self.clusters[i] for i in cluster_indices]
            new_clusters.append(self.generate_merged_cluster(clusters))
            clusters_to_remove.extend(clusters)
        for cluster in clusters_to_remove:
            self.clusters.remove(cluster)
        for new_cluster in new_clusters:
            self.clusters.append(new_cluster)
        for ind, cluster in enumerate(self.clusters):
            cluster.label = ind

        self.update_absolute_cluster_level_distances()
        return

    def generate_merged_cluster(self, clusters):
        """Takes a list of "cluster" objects and returns a new cluster that is
        merged version of given clusters.

        Parameters
        ----------
        clusters: A list of "Cluster" objects
        """

        new_cluster = Cluster(label=self.clusters[0].label,
                              members=[])
        for cluster in clusters:
            new_cluster.add_members(cluster.members)
        return new_cluster

    def update_absolute_cluster_level_distances(self):
        n_clusters = len(self.clusters)
        distances = np.zeros((n_clusters, n_clusters))
        for ind1, ind2 in itertools.combinations([i for i in range(n_clusters)], 2):
            dist = self.calc_cluster_level_absolute_distance(self.clusters[ind1],
                                                             self.clusters[ind2])
            distances[ind1, ind2] = dist

        distances = distances + distances.T
        self.cluster_level_absolut_distances = distances
        return

    def calc_cluster_level_absolute_distance(self, cluster1, cluster2):
        pair_wise_distances = list()
        for a in cluster1.members:
            for b in cluster2.members:
                dist = self.absolute_distances[a, b]
                pair_wise_distances.append(dist)
        return min(pair_wise_distances)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.clusters):
            res = self.clusters[self.n]
            self.n += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return len(self.clusters)


class Cluster:
    """Use this to make cluster objects.

    A cluster object is a container of data points that correspond to same cluster.

    Parameters
    ----------
    label: :int:
        Arbitrary label.
    members: :list of int:
        Give data index in original data array as member, e.g. : [5, 6] means 5th and 6th data point.
    """
    def __init__(self, label, members):
        self.label = int(label)
        self.members = members

    def add_members(self, members):
        """Use this method for appending new members to this cluster."""

        self.members.extend(members)

    def __len__(self):
        return len(self.members)


class PrimaryRO:
    def __init__(self, absolute_dist_metric='euclidean', threshold=5):
        self.absolute_dist_metric = absolute_dist_metric
        self.threshold = threshold

    def fit_predict(self, x):
        n_samples = x.shape[0]
        nbrs = NearestNeighbors(n_neighbors=n_samples,
                                algorithm='ball_tree',
                                metric=self.absolute_dist_metric).fit(x)
        _, order_lists = nbrs.kneighbors(x)
        ro_distances = self.calc_ro_distances(order_lists)
        clusters = self.do_clustering(ro_distances, self.threshold)
        labels = np.zeros(n_samples)
        for i, c in enumerate(clusters):
            labels[c] = i
        return labels

    def calc_ro_distances(self, order_lists):
        n_samples = order_lists.shape[0]
        ro_distances = np.zeros((n_samples, n_samples))
        indices = [i for i in range(n_samples)]

        # Calculating rank-order distances for each possible pair
        for i, j in itertools.combinations(indices, 2):
            ro_distances[i, j] = self.calc_dr(i, j, order_lists[i], order_lists[j])
        ro_distances = ro_distances + ro_distances.T
        return ro_distances

    @staticmethod
    def do_clustering(ro_distances, threshold):
        indices = [i for i in range(ro_distances.shape[0])]
        clusters = []
        while indices:
            ind = indices.pop()
            indices_copy = indices.copy()
            cluster = [ind]
            adjacents = np.where(ro_distances[ind] < threshold)[0]
            for i in indices_copy:
                if i in adjacents:
                    indices.remove(i)
                    cluster.append(i)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def calc_dr(ind1, ind2, ol1, ol2):
        def calc_asym_dist(ind, ol1, ol2):
            b_in_a = np.where(ol1 == ind)[0][0]
            d = 0
            for i in range(b_in_a):
                f = ol1[i]
                o = np.where(ol2 == f)[0][0]
                d += o
            return b_in_a, d

        o_a_b, d_1 = calc_asym_dist(ind2, ol1, ol2)
        o_b_a, d_2 = calc_asym_dist(ind1, ol2, ol1)
        d_r = (d_1 + d_2) / min(o_a_b, o_b_a)
        return d_r


class ChineseWhispersClustering:
    def __init__(self, n_iteration=5):
        self.n_iteration = n_iteration
        self.adjacency_mat_ = None
        self.labels_ = None

    def fit_predict(self, adjacency_mat):
        """Fits and returns labels for samples"""

        n_nodes = adjacency_mat.shape[0]
        indices = np.arange(n_nodes)
        labels_mat = np.arange(n_nodes)
        for _ in range(self.n_iteration):
            np.random.shuffle(indices)
            for ind in indices:
                weights = adjacency_mat[ind]
                winner_label = self._find_winner_label(weights, labels_mat)
                labels_mat[ind] = winner_label
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = labels_mat
        return labels_mat

    @staticmethod
    def _find_winner_label(node_weights, labels_mat):
        adjacent_nodes_indices = np.where(node_weights > 0)[0]
        adjacent_nodes_labels = labels_mat[adjacent_nodes_indices]
        unique_labels = np.unique(adjacent_nodes_labels)
        label_weights = np.zeros(len(unique_labels))
        for ind, label in enumerate(unique_labels):
            indices = np.where(adjacent_nodes_labels == label)
            weight = np.sum(node_weights[adjacent_nodes_indices[indices]])
            label_weights[ind] = weight
        winner_label = unique_labels[np.argmax(label_weights)]
        return winner_label


class ROGraph:
    def __init__(self, k, metric, algorithm):
        self.k = k
        self.metric = metric
        self.knn_algorithm = algorithm
        self.adjacency_mat_ = None

    @property
    def adjacency_mat(self):
        return self.adjacency_mat_

    def generate_graph(self, x):
        ordered_distances, order_lists = self._get_knns(x)
        pw_distances = self._generate_normalized_pw_distances(ordered_distances, order_lists)
        adjacency_mat = self._generate_adjacency_mat(pw_distances)
        return adjacency_mat

    def _get_knns(self, x):
        """Generates order lists and absolute distances of k-nearest-neighbors
            for each data point.
        """

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm=self.knn_algorithm, metric=self.metric).fit(x)
        ordered_absolute_distances, order_lists = nbrs.kneighbors(x)
        return ordered_absolute_distances, order_lists

    def _generate_normalized_pw_distances(self, ordered_distances, order_lists):
        n_samples = len(ordered_distances)
        combs = itertools.combinations([i for i in range(n_samples)], 2)
        pw_distances = np.zeros((n_samples, n_samples))
        for ind1, ind2 in combs:
            order_list_1, order_list_2 = order_lists[ind1], order_lists[ind2]
            pw_dist = self._calc_pw_dist(ind1, ind2, order_list_1, order_list_2)
            pw_distances[ind1, ind2] = pw_dist
        pw_distances = pw_distances / np.max(pw_distances)
        pw_distances = pw_distances + pw_distances.T
        return pw_distances

    def _generate_adjacency_mat(self, pw_distances):
        adjacency_mat = self._dist2adjacency(pw_distances)
        self.adjacency_mat_ = adjacency_mat
        return adjacency_mat

    @staticmethod
    def _dist2adjacency(distances):
        mask_mat = np.zeros(distances.shape)
        mask_mat[np.where(distances > 0)] = 1
        adjacency_mat = (1 - distances) * mask_mat
        return adjacency_mat

    def _calc_pw_dist(self, ind_a, ind_b, order_list_a, order_list_b):
        pw_dist = 0.0
        if np.any(order_list_a == order_list_b):
            order_b_in_a, order_a_in_b = self._calc_orders(ind_a, ind_b, order_list_a, order_list_b)
            d_m_ab = self._calc_dm(order_list_a, order_list_b, order_b_in_a)
            d_m_ba = self._calc_dm(order_list_b, order_list_a, order_a_in_b)
            pw_dist = (d_m_ab + d_m_ba) / min(order_a_in_b, order_b_in_a)
        return pw_dist

    def _calc_orders(self, ind_a, ind_b, order_list_a, order_list_b):
        order_b_in_a = np.where(order_list_a == ind_b)[0]
        if not order_b_in_a.size:
            order_b_in_a = self.k
        else:
            order_b_in_a = order_b_in_a[0]
        order_a_in_b = np.where(order_list_b == ind_a)[0]
        if not order_a_in_b.size:
            order_a_in_b = self.k
        else:
            order_a_in_b = order_a_in_b[0]
        return order_b_in_a, order_a_in_b

    def _calc_dm(self,
                 order_list_a,
                 order_list_b,
                 order_b_in_a):
        dist = 0
        for i in range(min(self.k, order_b_in_a)):
            sample_index = order_list_a[i]
            if np.any(order_list_b == sample_index):
                dist += 1 / self.k
            else:
                dist += 1
        return dist
