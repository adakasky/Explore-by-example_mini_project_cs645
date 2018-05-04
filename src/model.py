from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

np.random.seed(0)

from sklearn.cluster import KMeans

from src.util import load_data


class ClusterTreeNode(object):
    def __init__(self, data, rel=False, k=4):
        self.k = k
        self.data = data
        self.rel = rel
        self.relevant = []
        self.irrelevant = []
    
    def sample(self):
        self.cluster = KMeans(n_clusters=self.k, random_state=0)
        self.cluster.fit(self.data)
        pred = self.cluster.labels_
        centroids = self.cluster.cluster_centers_
        self.clustering = np.array([np.array(data)[np.where(pred == i)] for i in range(self.k)])
        self.boundaries = [{'min': self.clustering[i].min(axis=0),
                            'max': self.clustering[i].max(axis=0)} for i in range(self.k)]
        self.samples = np.array([self.clustering[i][np.argmin(np.sqrt(np.sum(
            (self.clustering[i] - centroids[i]) ** 2, axis=1)))] for i in range(self.k)])
    
    def split(self):
        relevance = query(self.samples)
        self.relevant.extend([ClusterTreeRelNode(c, rel=True, k=4)
                              for i, c in enumerate(self.clustering) if relevance[i]])
        self.irrelevant.extend([ClusterTreeIrrNode(c, rel=False, k=4)
                                for i, c in enumerate(self.clustering) if not relevance[i]])


class ClusterTreeRelNode(ClusterTreeNode):
    def __init__(self, data, rel=True, k=4):
        super(ClusterTreeRelNode, self).__init__(data, rel, k)
    
    def sample(self):  # sample around boundaries and shrink the boundaries
        boundary_idx = np.hstack([self.data.argmin(axis=0), self.data.argmax(axis=0)])
        self.samples = self.data[boundary_idx]
        relevance = query(self.samples)
        del_idx = boundary_idx[np.where(~relevance)]
        self.data = np.delete(self.data, del_idx, axis=0)
        self.boundaries = [{'min': self.data.min(axis=0), 'max': self.data.max(axis=0)}]


class ClusterTreeIrrNode(ClusterTreeNode):
    def __init__(self, data, rel=False, k=4):
        super(ClusterTreeIrrNode, self).__init__(data, rel, k)
    
    def split(self):
        relevance = query(self.samples)
        self.relevant.extend([ClusterTreeFNNode(c, rel=True, k=self.k)
                              for i, c in enumerate(self.clustering) if relevance[i]])
        self.irrelevant.extend([ClusterTreeIrrNode(c, rel=False, k=self.k)
                                for i, c in enumerate(self.clustering) if not relevance[i]])


class ClusterTreeFNNode(ClusterTreeNode):
    def __init__(self, data, rel=True, k=4):
        super(ClusterTreeFNNode, self).__init__(data, rel, k)
    
    def sample(self):  # sample around centroid and expand the boundaries
        pass


def query(data):
    return (data[:, 0] > 39) & (data[:, 0] < 77) & (data[:, 1] > 25) & (data[:, 1] < 56)


if __name__ == '__main__':
    data_path = "../data/sdss_100k.csv.gz"
    columns = ['rowc', 'colc', 'ra', 'field', 'fieldid', 'dec']
    data = np.array(load_data(data_path, columns))
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)) * 100
    
    ground_truth = data[query(data)]
    root = ClusterTreeNode(data, k=4)
