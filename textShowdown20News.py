import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
import time
from scipy import sparse
from numpy import linalg as LA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity



class Tfidf():
	def __init__(self):
		self.sublinear_tf= True

	# def getTf():


	# def fit():

	def fit_transform(self,X):
		
		d, t = X.shape
		tf = np.zeros(X.shape)
		non_zero = X.nonzero()
		sparseRows = non_zero[0]
		sparseColumns = non_zero[1]

		for i in range(len(sparseRows)):
			row = sparseRows[i]
			col = sparseColumns[i]
			tf[row][col] = np.log(X[row,col]) +1

		

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
		tfidf = tf

		for i in range(d):
			for j in range(t):
				if tf[i][j] != 0:
					tfidf[i][j] = tf[i][j] * self.idf[j]

		#normalization
		scalar = np.zeros(d)
		for i in range(d):
			scalar[i] = LA.norm(tfidf[i][:])
			tfidf[i][:] = np.true_divide(tfidf[i][:], scalar[i])


		return sparse.csr_matrix(tfidf)


	def transform(self, X):
		d, t = X.shape
		tf = np.zeros(X.shape)
		non_zero = X.nonzero()
		sparseRows = non_zero[0]
		sparseColumns = non_zero[1]

		for i in range(len(sparseRows)):
			row = sparseRows[i]
			col = sparseColumns[i]
			tf[row][col] = np.log(X[row,col]) +1

		tfidf = tf
		for i in range(d):
			for j in range(t):
				if tf[i][j] != 0:
					tfidf[i][j] = tf[i][j] * self.idf[j]
		#normalization
		scalar = np.zeros(d)
		for i in range(d):
			scalar[i] = LA.norm(tfidf[i][:])
			tfidf[i][:] = np.true_divide(tfidf[i][:], scalar[i])

		return sparse.csr_matrix(tfidf)



if __name__ == '__main__':
	
	twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
	
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	tfidf_transformer = TfidfTransformer(norm=u'l2',sublinear_tf= True)
	# tfidf_transformer = Tfidf()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	start = time.time()
	clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)	
	
	#clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf,twenty_train.target)
	# nb_clf = Pipeline([('vect', CountVectorizer()),
 #                     ('tfidf', TfidfTransformer(norm=u'l2',sublinear_tf= True)),
 #                     ('clf', MultinomialNB()),
	# ])

	
	
	# nb_clf = nb_clf.fit(twenty_train.data, twenty_train.target)
	end = time.time()
	test_trainingTime = end-start


	#testing data
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
	docs_test = twenty_test.data
	X_new_counts = count_vect.transform(docs_test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	nb_predicted = clf.predict(X_new_tfidf)
	# nb_predicted = clf.predict(docs_test)
	
	
	# accuracy = np.mean(nb_predicted == twenty_test.target)
	# print accuracy
	
	# print "nb\n"
	
	# test_recall = metrics.recall_score(twenty_test.target, nb_predicted)
	# test_precision = metrics.precision_score(twenty_test.target, nb_predicted)

	test_accuracy = metrics.accuracy_score(twenty_test.target, nb_predicted)
	test_precision, test_recall, fscore, support= precision_recall_fscore_support(twenty_test.target, nb_predicted, average='macro')	

	X_train_new_counts = count_vect.transform(twenty_train.data)
	X_train_new_tfidf = tfidf_transformer.transform(X_train_new_counts)
	nb_predicted_train = clf.predict(X_train_new_tfidf)

	# train_recall = metrics.recall_score(twenty_train.target, nb_predicted_train)
	# train_precision = metrics.precision_score(twenty_train.target, nb_predicted_train)
	train_accuracy = metrics.accuracy_score(twenty_train.target, nb_predicted_train)
	train_precision, train_recall, fscore, support= precision_recall_fscore_support(twenty_train.target, nb_predicted_train, average='macro')	
	'''
		calculation of training part
	'''
	#split n/10
	# n = len(twenty_train)
	# split = n/10
	# twenty_test = twenty_train[(-split):]
	# twenty_train = twenty_train[:(n-split)]

	# start = time.time()
	# nb_clf = nb_clf.fit(twenty_train.data, twenty_train.target)
	# end = time.time()
	# train_trainingTime = end-start


	# #training data
	
	# docs_test = twenty_test.data
	# nb_predicted = nb_clf.predict(docs_test)

	# train_recall = metrics.recall_score(twenty_test.target, nb_predicted)
	# train_precision = metrics.precision_score(twenty_test.target, nb_predicted)
	# train_accuracy = metrics.accuracy_score(twenty_test.target, nb_predicted)

	test_dict = {'accuracy': [test_accuracy,train_accuracy], 'recall':[test_recall,train_recall], 'precision':[test_precision,train_precision], 'training_time': [test_trainingTime,test_trainingTime]}
	

	print(pd.DataFrame(test_dict, index = ['nb_test', 'nb_train']))



	'''
		svm part
	'''

	start = time.time()
	# clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf,twenty_train.target)
	clf = SVC(kernel = cosine_similarity).fit(X_train_tfidf,twenty_train.target)
	
	# svm_clf = svm_clf.fit(twenty_train.data, twenty_train.target)
	end = time.time()
	test_trainingTime = end-start
	
	#testing data
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
	docs_test = twenty_test.data
	X_new_counts = count_vect.transform(docs_test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	svm_predicted = clf.predict(X_new_tfidf)
	
	# svm_predicted = svm_clf.predict(docs_test)

	# test_recall = metrics.recall_score(twenty_test.target, svm_predicted)
	# test_precision = metrics.precision_score(twenty_test.target, svm_predicted)
	test_accuracy = metrics.accuracy_score(twenty_test.target, svm_predicted)
	test_precision, test_recall, fscore, support= precision_recall_fscore_support(twenty_test.target, svm_predicted, average='macro')	

	X_train_new_counts = count_vect.transform(twenty_train.data)
	X_train_new_tfidf = tfidf_transformer.transform(X_train_new_counts)
	svm_predicted_train = clf.predict(X_train_new_tfidf)

	# train_recall = metrics.recall_score(twenty_train.target, svm_predicted_train)
	# train_precision = metrics.precision_score(twenty_train.target, svm_predicted_train)
	train_accuracy = metrics.accuracy_score(twenty_train.target, svm_predicted_train)
	train_precision, train_recall, fscore, support= precision_recall_fscore_support(twenty_train.target, svm_predicted_train, average='macro')	
	# print "svm\n"
	# print(metrics.classification_report(twenty_test.target, svm_predicted, target_names=twenty_test.target_names))

	test_dict = {'accuracy': [test_accuracy,train_accuracy], 'recall':[test_recall,train_recall], 'precision':[test_precision,train_precision], 'training_time': [test_trainingTime,test_trainingTime]}
	
	print(pd.DataFrame(test_dict, index = ['svm_test', 'svm_train']))
