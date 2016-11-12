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
if __name__ == '__main__':

	twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	tfidf_transformer = TfidfTransformer(norm=u'l2',sublinear_tf= True)
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)	
	
	#clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf,twenty_train.target)
	nb_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(norm=u'l2',sublinear_tf= True)),
                     ('clf', MultinomialNB()),
	])

	
	start = time.time()
	nb_clf = nb_clf.fit(twenty_train.data, twenty_train.target)
	end = time.time()
	test_trainingTime = end-start


	#testing data
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
	docs_test = twenty_test.data
	# X_new_counts = count_vect.transform(docs_test)
	# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	# predicted = clf.predict(X_new_tfidf)
	nb_predicted = nb_clf.predict(docs_test)
	
	
	# accuracy = np.mean(nb_predicted == twenty_test.target)
	# print accuracy
	
	# print "nb\n"

	test_recall = metrics.recall_score(twenty_test.target, nb_predicted)
	test_precision = metrics.precision_score(twenty_test.target, nb_predicted)
	test_accuracy = metrics.accuracy_score(twenty_test.target, nb_predicted)

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

	test_dict = {'accuracy': [test_accuracy,test_accuracy], 'recall':[test_recall,test_recall], 'precision':[test_precision,test_precision], 'training_time': [test_trainingTime,test_trainingTime]}
	

	print(pd.DataFrame(test_dict, index = ['nb_test', 'nb_train']))



	'''
		svm part
	'''

	svm_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(norm=u'l2',sublinear_tf= True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf,twenty_train.target)),
	])

	start = time.time()
	svm_clf = svm_clf.fit(twenty_train.data, twenty_train.target)
	end = time.time()
	test_trainingTime = end-start
	
	#testing data
	twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
	docs_test = twenty_test.data
	# X_new_counts = count_vect.transform(docs_test)
	# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	# predicted = clf.predict(X_new_tfidf)
	
	svm_predicted = svm_clf.predict(docs_test)

	test_recall = metrics.recall_score(twenty_test.target, svm_predicted)
	test_precision = metrics.precision_score(twenty_test.target, svm_predicted)
	test_accuracy = metrics.accuracy_score(twenty_test.target, svm_predicted)

	# print "svm\n"
	# print(metrics.classification_report(twenty_test.target, svm_predicted, target_names=twenty_test.target_names))

	test_dict = {'accuracy': [test_accuracy,test_accuracy], 'recall':[test_recall,test_recall], 'precision':[test_precision,test_precision], 'training_time': [test_trainingTime,test_trainingTime]}
	
	print(pd.DataFrame(test_dict, index = ['svm_test', 'svm_train']))
