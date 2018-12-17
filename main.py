#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Desafio KDD 2009 '''

# Imports
import argparse
import numpy as np
import pandas as pd

# Classifiers
from xgboost import XGBClassifier

# Metrics
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 
def arg_passing():
	''' Argument Parsing '''
	parser = argparse.ArgumentParser(description='KDD 2009 Challenge')

	# Main arguments
	parser.add_argument('-d', nargs=1, type=str, default=["data/orange_small_train.data"], help='Data file')
	parser.add_argument('-l', nargs=1, type=str, required=True, help='Labels file')
	parser.add_argument('-s', nargs=1, type=str, default=["\t"], help='Separator')
	parser.add_argument('-k', nargs=1, type=float, default=[10], help='Number of folds / Test Split Percentange')

	parser.add_argument('-p', nargs=1, type=float, help='Exclude nominal with x null percentange columns')
	parser.add_argument('-e', action='store_true', help='Exclude nominal columns')

	# XGBoost arguments
	parser.add_argument('--max_depth', nargs=1, type=int, default=[3], help='XGBoost.max_depth')
	parser.add_argument('--n_estimators', nargs=1, type=int, default=[100], help='XGBoost.n_estimators')
	parser.add_argument('--subsample', nargs=1, type=float, default=[1], help='XGBoost.subsample')
	parser.add_argument('--gamma', nargs=1, type=float, default=[0], help='XGBoost.gamma')

	#
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
		print('Dropped all nominal columns')
	else:
		obj_columns = X_data.select_dtypes(include=['object']).columns
		X_data[obj_columns] = X_data[obj_columns].astype('category').apply(lambda x: x.cat.codes)

		if args.p is not None:
			total = len(X_data)
			drop_cols = []

			for i in obj_columns:
				if len(X_data[X_data[i] == 0]) / total > args.p[0]:
					drop_cols.append(i)

			X_data.drop(drop_cols, axis=1, inplace=True)
			print('Dropped {0} nominal columns. Percentage {1}'.format(len(drop_cols), args.p[0]))

	# Classifier
	classifier = XGBClassifier(
		max_depth=args.max_depth[0], 
		n_estimators=args.n_estimators[0], 
		subsample=args.subsample[0],
		gamma=args.gamma[0])

	# Split
	if split < 1.0:
		print('Running {0} split evaluation'.format(split))

		X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=split, random_state=1)
		classifier.fit(X_train, y_train.values.ravel())
		
		score = classifier.score(X_test, y_test)
		y_hat = classifier.predict(X_test)

		print('Split {0}, Score: {1}, ROC_AUC: {2}'.format(
			split, score, metrics.roc_auc_score(y_test, y_hat)))

	# K-fold
	else:
		print('Running {0} fold(s) cross-validation'.format(int(split)))

		k_fold = KFold(int(split))
		rocs, scores = [], []

		for k, (train, test) in enumerate(k_fold.split(X_data, y_data)):
			X_train, y_train = X_data.iloc[train], y_data.iloc[train]
			X_test, y_test = X_data.iloc[test], y_data.iloc[test]

			classifier.fit(X_train, y_train.values.ravel())

			score = classifier.score(X_test, y_test)
			y_hat = classifier.predict(X_test)
			roc = metrics.roc_auc_score(y_test, y_hat)

			scores.append(score)
			rocs.append(roc)

			print('Fold {0}, Score: {1}, ROC_AUC: {2}'. format(
				k + 1, score, roc))

		print('-')
		print('Score: {0}, ROC_AUC: {1}'.format(np.mean(score), np.mean(rocs)))
