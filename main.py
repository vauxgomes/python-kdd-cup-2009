#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Desafio KDD 2009 '''

# Imports
import argparse
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# Classifiers
from xgboost import XGBClassifier

# Metrics
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV

# 
def arg_passing():
	''' Argument Parsing '''
	parser = argparse.ArgumentParser(description='KDD 2009 Challenge')

	# Main arguments
	parser.add_argument('-d', nargs=1, type=str, default=["data/orange_small_train.data"], help='Data file')
	parser.add_argument('-l', nargs=1, type=str, required=True, help='Labels file')
	parser.add_argument('-s', nargs=1, type=str, default=["\t"], help='Separator')
	parser.add_argument('-k', nargs=1, type=float, default=[10], help='Number of folds / Test Split Percentange')

	parser.add_argument('-e', action='store_true', help='Exclude object columns')
	return parser.parse_args()

#
if __name__ == '__main__':
	''' Main '''
	args = arg_passing()

	# Parameters
	data = args.d[0]
	labels = args.l[0]
	sep = args.s[0]
	split = args.k[0]

	# Load data
	X_data = pd.read_table(data, sep=sep)
	y_data = pd.read_table(labels, sep=sep, header=None)

	# Data pre-processing
	X_data.fillna(0, inplace=True)

	if args.e:
		X_data = X_data.select_dtypes(exclude=['object'])
	else:
		obj_columns = X_data.select_dtypes(include=['object']).columns
		X_data[obj_columns] = X_data[obj_columns].astype('category').apply(lambda x: x.cat.codes)

	# Classifier
	classifier = XGBClassifier()

	# Split
	if split < 1.0:
		X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split, random_state=1)

		classifier.fit(X_train, y_train.values.ravel())
		
		score = classifier.score(X_test, y_test)
		y_hat = classifier.predict(X_test)
		fp, tp, _ = metrics.roc_curve(y_test, y_hat, pos_label=1)

		print('Split {0}, Score: {1}, AUC: {2}'.format(
			split, score, metrics.auc(fp, tp)))

	# K-fold
	else:
		k_fold = KFold(int(split))
		aucs, scores = [], []

		for k, (train, test) in enumerate(k_fold.split(X_data, y_data)):
			X_train, y_train = X_data.iloc[train], y_data.iloc[train]
			X_test, y_test = X_data.iloc[test], y_data.iloc[test]

			classifier.fit(X_train, y_train.values.ravel())

			y_hat = classifier.predict(X_test)
			fp, tp, _ = metrics.roc_curve(y_test, y_hat, pos_label=1)
			score = classifier.score(X_test, y_test)
			auc = metrics.auc(fp, tp)

			scores.append(score)
			aucs.append(auc)

			print('Fold {0}, Score: {1}, AUC: {2}'. format(
				k, score, auc))

		print('-')
		print('Score: {0}, AUC: {1}'.format(np.array(score).mean(), np.array(aucs).mean()))
