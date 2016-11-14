import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from scipy import sparse
from sklearn import metrics
from sklearn.pipeline import Pipeline
from numpy import linalg as LA
from copy import deepcopy
class Tfidf():
	def __init__(self):
		self.sublinear_tf= True

	# def getTf():


	# def fit():

	def fit_transform(self,X):
		
		d, t = X.shape

		tf = deepcopy(X)
		non_zero = X.nonzero()
		sparseRows = non_zero[0]
		sparseColumns = non_zero[1]

		for i in range(len(sparseRows)):
			row = sparseRows[i]																																																			
			col = sparseColumns[i]
			tf[row,col] = np.log(X[row,col]) +1

		print t,d

		self.idf = np.zeros(t)
		totalSetOfDocuments = d
		for i in range(t):
			subsetOfDocuments = len(sparseColumns[sparseColumns == i])
			if totalSetOfDocuments < subsetOfDocuments+1:
				self.idf[i] = 0
			else:
				self.idf[i] = np.log(totalSetOfDocuments / (subsetOfDocuments +1.0) )
			# if self.idf[i] < 0:
			# 	print totalSetOfDocuments, subsetOfDocuments
		# print self.idf[self.idf < 0]
		tfidf = deepcopy(tf)
		non_zero = tf.nonzero()
		sparseRows = non_zero[0]
		sparseColumns = non_zero[1]
		for i in range(len(sparseRows)):
			row = sparseRows[i]
			col = sparseColumns[i]
			tfidf[row,col] = tf[row,col] * self.idf[col]

		#normalization
		startIndex = 0
		current = sparseRows[0]
		# endIndex = 0
		for i in range(len(sparseRows)):
			nxt = sparseRows[i]
			if nxt != current: #i now marks the start of a new document
				endIndex = i #from start to end-1 belongs to the same document
				squaredSum = 0.0

				for j in range(startIndex, endIndex):
					row = sparseRows[j]
					col = sparseColumns[j]
					squaredSum += tfidf[row,col]**2

				vector_length = squaredSum ** 0.5
				for j in range(startIndex, endIndex):
					row = sparseRows[j]
					col = sparseColumns[j]
					tfidf[row,col] /= vector_length				
				if i != len(sparseRows)-1:
					startIndex = i
					current = sparseRows[startIndex]

		return tfidf
		# scalar = np.zeros(d)
		# for i in range(d):
		# 	scalar[i] = LA.norm(tfidf[i][:])
		# 	tfidf[i][:] = np.true_divide(tfidf[i][:], scalar[i])


		# return sparse.csr_matrix(tfidf)
		

	def transform(self, X):
		# d, t = X.shape
		# tf = np.zeros(X.shape)
		# non_zero = X.nonzero()
		# sparseRows = non_zero[0]
		# sparseColumns = non_zero[1]

		# for i in range(len(sparseRows)):
		# 	row = sparseRows[i]
		# 	col = sparseColumns[i]
		# 	tf[row][col] = np.log(X[row,col]) +1

		# tfidf = np.zeros(tf.shape)
		# for i in range(d):
		# 	for j in range(t):
		# 		if tf[i][j] != 0:
		# 			tfidf[i][j] = tf[i][j] * self.idf[j]
		# #normalization
		# scalar = np.zeros(d)
		# for i in range(d):
		# 	scalar[i] = LA.norm(tfidf[i][:])
		# 	tfidf[i][:] = np.true_divide(tfidf[i][:], scalar[i])

		# # return sparse.csr_matrix(tfidf)
		# return tfidf

		d, t = X.shape

		tf = deepcopy(X)
		non_zero = X.nonzero()
		sparseRows = non_zero[0]
		sparseColumns = non_zero[1]

		for i in range(len(sparseRows)):
			row = sparseRows[i]
			col = sparseColumns[i]
			tf[row,col] = np.log(X[row,col]) +1

		print t,d

		tfidf = deepcopy(tf)
		non_zero = tf.nonzero()
		sparseRows = non_zero[0]
		sparseColumns = non_zero[1]
		for i in range(len(sparseRows)):
			row = sparseRows[i]
			col = sparseColumns[i]
			tfidf[row,col] = tf[row,col] * self.idf[col]

		#normalization
		startIndex = 0
		current = sparseRows[0]
		# endIndex = 0
		for i in range(len(sparseRows)):
			nxt = sparseRows[i]
			if nxt != current: #i now marks the start of a new document
				endIndex = i #from start to end-1 belongs to the same document
				squaredSum = 0.0

				for j in range(startIndex, endIndex):
					row = sparseRows[j]
					col = sparseColumns[j	]
					squaredSum += tfidf[row,col]**2

				vector_length = squaredSum ** 0.5
				for j in range(startIndex, endIndex):
					row = sparseRows[j]
					col = sparseColumns[j]
					tfidf[row,col] /= vector_length				
				if i != len(sparseRows)-1:
					startIndex = i
					current = sparseRows[startIndex]

		return tfidf

if __name__ == '__main__':
	categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
	# twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories = categories, random_state=42)
	# count_vect = CountVectorizer()
	# X_train_counts = count_vect.fit_transform(twenty_train.data)
	# tfidf_transformer = TfidfTransformer(norm=u'l2',sublinear_tf= True)
	# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)	

	# twenty_test = fetch_20newsgroups(subset='test',
 #    categories=categories, shuffle=True, random_state=42)

	# X_new_counts = count_vect.transform(twenty_test)
	# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	
	# predicted = clf.predict(X_new_tfidf)
	# print(np.mean(predicted == twenty_test.target))
	twenty_train = fetch_20newsgroups(subset='train', categories = categories, shuffle=True, random_state=42)
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	# tfidf_transformer = TfidfTransformer(norm=u'l2',sublinear_tf= True)
	tfidf_transformer = Tfidf()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)	
	# clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf,twenty_train.target)
	
	nb_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(norm=u'l2',sublinear_tf= True)),
                     ('clf', MultinomialNB()),
	])

	# clf = clf.fit(twenty_train.data, twenty_train.target)
	
	# nb_clf = nb_clf.fit(twenty_train.data, twenty_train.target)
	


	#testing data
	twenty_test = fetch_20newsgroups(subset='test', categories = categories, shuffle=True, random_state=42)
	docs_test = twenty_test.data
	X_new_counts = count_vect.transform(docs_test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	predicted = clf.predict(X_new_tfidf)
	# nb_predicted = clf.predict(docs_test)
	
	
	# accuracy = np.mean(nb_predicted == twenty_test.target)
	# print accuracy
	
	# print "nb\n"

	# test_recall = metrics.recall_score(twenty_test.target, nb_predicted)
	# test_precision = metrics.precision_score(twenty_test.target, nb_predicted)
	test_accuracy = metrics.accuracy_score(twenty_test.target, predicted)
	print test_accuracy

