#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Desafio KDD 2009 '''

# Imports
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Gsearch
def gsearch(X_data, y_data):
	param_grid = {
		'max_depth': [3, 5],
		'n_estimators': [50, 100, 300],
		'nthread': [8],
		'subsample': [0.7, 0.8, 0.9, 1.0]
	}

	grid = GridSearchCV(XGBClassifier(), param_grid, refit=True, verbose=3, scoring='roc_auc', n_jobs=4)
	grid.fit(X_data, y_data.values.ravel())
	
	print('Params', grid.best_params_)
	print('Score', grid.best_score_)

	return grid

# Main
# Load data & Processing
X_data = pd.read_table('data/orange_small_train.data', sep='\t')
X_data.fillna(0, inplace=True)
obj_columns = X_data.select_dtypes(include=['object']).columns
X_data[obj_columns] = X_data[obj_columns].astype('category').apply(lambda x: x.cat.codes)

#
y_data = pd.read_table('data/labels/orange_small_train_churn.labels', sep='\t', header=None)
grid1 = gsearch(X_data, y_data)

#
y_data = pd.read_table('data/labels/orange_small_train_appetency.labels', sep='\t', header=None)
grid2 = gsearch(X_data, y_data)

#
y_data = pd.read_table('data/labels/orange_small_train_upselling.labels', sep='\t', header=None)
grid3 = gsearch(X_data, y_data)
