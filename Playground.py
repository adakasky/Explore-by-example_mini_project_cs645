
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import psycopg2
from sklearn import tree
from sklearn.cluster import KMeans

normalize = lambda a: (a - a.min()) / (a.max() - a.min()) * 100

class Sampler:

	def __init__(self, database, user, query, keys):

		self.keys = keys
		self.conn = psycopg2.connect("dbname=" + database +  " user="+user)
		self.cur = self.conn.cursor()
		self.cur.execute(query)
		self.conn.commit()
		self.cache = np.array(self.cur.fetchall())

	def data_setup(self, false_query, size=3):
		true_sample = self.cache[np.random.randint(len(self.cache), size=size)]
		self.cur.execute(false_query)
		self.false_result = np.array(self.cur.fetchall())
		self.conn.commit()
		false_sample = self.false_result[np.random.randint(len(self.false_result), size=size)]
		return self.cache, true_sample, false_sample

	'''
	@clf: a decision tree classifier 
	@training_data: dataset to be sampled
	@k: number of iteration
	@y: distance 
	'''
	# sample(tree, false_negative, clusters, y)
	def sample(self, clf, training_data, k=6, y=1, f=10):

		prediction = clf.predict(training_data)
		kmean_training = training_data[np.where(prediction == 0)]
		if len(kmean_training) == 0:
			return 'False'
		kmean = KMeans(n_clusters=k, random_state=0).fit(kmean_training)
		
		# Map each tranining point to corresponding cluster
		self.clusters = {l:[] for l in kmean.labels_}
		# In each cluster, find out boundaries of each dimension
		self.sample_info = {label:{dim:{'max':0, 'min':0} for dim in range(k)} for label in kmean.labels_}
		
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
				query += keys[a] + ' between ' + str(self.sample_info[k][a]['min']) + ' and ' + str(self.sample_info[k][a]['max']) + ' and '
			Q.append(query[:-4] + ' limit ' + str(f))
		return Q

true_query = "select * from photoobjall where rowc > 39 and rowc < 77 and colc > 25 and colc < 56"
false_query = "select * from photoobjall where rowc < 39 and colc > 25"
keys = {0:'rowc',1:'colc',2:'ra', 3:'field',4:'fieldid',5:'dec'}

sampler = Sampler('zitao', 'zitao', true_query, keys)

#data = pd.read_csv('sdss_100k.csv')
#clean_data = data.loc[:,['rowc','colc','ra','field','fieldid','dec']]

cache, true_sample, false_sample = sampler.data_setup(false_query, 3)

training = np.vstack((true_sample, false_sample))
labels = np.array([1, 1, 1, 0, 0, 0])
clf = tree.DecisionTreeClassifier(random_state = 0)
clf.fit(training, labels)
training_data = cache[np.random.randint(len(cache), size=1000)]

Q = sampler.sample(clf, training_data)
print(Q)


