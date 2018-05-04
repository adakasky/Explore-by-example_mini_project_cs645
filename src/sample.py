# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cluster import KMeans


class Sampler:
    # query format: {'rowc': (lower, upper), 'colc': (lower, upper) ....}
    # dataset.loc[(self.dataset['rowc'] > lower) & (self.dataset['rowc'] <= upper) & (self.dataset['colc'] > lower) & (self.dataset['colc'] <= upper)] ...
    def __init__(self, dataset, keys, normalize=False):
        # self.query = query
        self.dataset = dataset
        self.keys = keys
        self.f = lambda a: (a - a.min()) / (a.max() - a.min()) * 100
        if normalize:
            self.dataset = self.dataset.apply(self.f)
    
    def query_eval(self, query):
        self.parsed = ''
        for k, v in query.items():
            self.parsed += '(' + str(k) + '>' + str(v[0]) + ')' + '&' + '(' + str(k) + '<' + str(v[1]) + ')' + '&'
        return self.dataset.query(self.parsed[:-1])
    
    '''
    @clf: a decision tree classifier
    @training_data: dataset to be sampled
    @k: number of cluster
    @y: distance
    '''
    
    # sample(tree, false_negative, clusters, y)
    def sample(self, clf, training_data, k=6, y=1, f=10):
        
        prediction = clf.predict(training_data)
        kmean_training = training_data[np.where(prediction == 0)]
        if len(kmean_training) == 0:
            print(prediction)
            return 'False'
        kmean = KMeans(n_clusters=k, random_state=0).fit(kmean_training)
        
        # Map each tranining point to corresponding cluster
        self.clusters = {l: [] for l in kmean.labels_}
        # In each cluster, find out boundaries of each dimension
        self.sample_info = {label: {dim: {'max': 0, 'min': 0} for dim in range(k)} for label in kmean.labels_}
        
        for l in range(len(kmean.labels_)):
            self.clusters[kmean.labels_[l]].append(kmean_training[l])
        
        for l in self.sample_info.keys():
            for i in range(k):
                self.sample_info[kmean.labels_[l]][i]['max'] = np.max(self.clusters[kmean.labels_[l]], axis=0)[i]
                self.sample_info[kmean.labels_[l]][i]['min'] = np.min(self.clusters[kmean.labels_[l]], axis=0)[i]
        
        Q = []
        # For each attributes ...
        for k in self.sample_info.keys():
            # For each dimension in the keys ...
            query = 'where '
            for a in range(len(self.sample_info[k])):
                query += keys[a] + ' between ' + str(self.sample_info[k][a]['min']) + ' and ' + str(
                    self.sample_info[k][a]['max']) + ' and '
            Q.append(query[:-4] + ' limit ' + str(f))
        return Q


# true_query = "select * from photoobjall where rowc > 39 and rowc < 77 and colc > 25 and colc < 56"
# false_query = "select * from photoobjall where rowc < 39 and colc > 25"
keys = {0: 'rowc', 1: 'colc', 2: 'ra', 3: 'field', 4: 'fieldid', 5: 'dec'}
data = pd.read_csv('sdss_100k.csv')
clean_data = data.loc[:, ['rowc', 'colc', 'ra', 'field', 'fieldid', 'dec']]

sampler = Sampler(clean_data, keys, normalize=True)

cache = sampler.query_eval({'colc': (25, 77), 'ra': (25.395, 35)})
true_sample = cache.sample(n=3)
false_sample = sampler.query_eval({'rowc': (0, 25)}).sample(n=3)

training = np.vstack((true_sample, false_sample))
labels = np.array([1, 1, 1, 0, 0, 0])
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(training, labels)
training_data = np.array(cache.sample(1000))

Q = sampler.sample(clf, training_data)
print(Q)
