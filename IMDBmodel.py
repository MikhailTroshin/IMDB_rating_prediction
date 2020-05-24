# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:05:08 2020

@author: Mihul

Contains a model for IMDB rating prediction
"""

# importing submodules
import IMDBpreprocessing_funcs as prep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data
dataset = pd.read_csv('dataset_00_19.csv')


# _____________________ preparing the dataset_________________

dataset1 = dataset.drop(columns = ['Description'])

# Certificate column
dataset1['Certificate'].value_counts()
# Getting rid of rubbish categories (make them all Unrated)
cathegories = ['G', 'PG', 'PG-13', 'R', 'NC-17'] # possible movie categories 
for i in range(len(dataset1['Certificate'])):
    if dataset1.at[i, 'Certificate'] not in cathegories:
        dataset1.at[i, 'Certificate'] = 'Unrated'
# making dummy-variable table
certificate_dummy = pd.get_dummies(dataset1['Certificate']) # df of uint8 - OK


# Runtime column
from sklearn.preprocessing import StandardScaler
sc_runtime = StandardScaler()
runtime_raw = dataset1.iloc[:, 1].values # for trees ansambles
runtime = sc_runtime.fit_transform(runtime_raw.reshape(-1,1)) # array
runtime = pd.Series(runtime.flatten(), name = 'Runtime') # Series of float


# Genre
genre_dummy = prep.get_multidummies(dataset1['Genre'], sep = ',', n_values = 0, drop_first = True) # df
genre_dummy = genre_dummy.astype('uint8') # df of uint8 - OK


# Directors
directors_dummy = pd.get_dummies(dataset1['Directors']) # df of uint8 - OK


# Stars
stars_dummy = prep.get_multidummies(dataset1['Stars'], sep = ',', n_values = 0, drop_first = True)
# get all actors with n_films = 1
loosers = stars_dummy.sum(axis = 0) == 1
loosers = loosers[loosers == True].index #list of over 3000 actors - huge potential for reducing
# before reduction: 4900
stars_dummy = prep.reduce_by_min_count(stars_dummy, loosers) # df
# after reduction: 1871
# write stars_dummy to .csv
stars_dummy.to_csv('stars_dummy.csv', index = False)
stars_dummy = stars_dummy.astype('uint8') # df of uint8 - OK


# Rating - target
sc_rating = StandardScaler()
rating_raw = dataset1.iloc[:, -1].values  # for trees ansambles
rating = sc_rating.fit_transform(rating_raw.reshape(-1,1)) # array


# Making X-matrix
to_stack = [certificate_dummy, runtime, genre_dummy, directors_dummy, stars_dummy]
X1 = pd.concat(to_stack, axis = 1)
y1 = rating

to_stack_raw = [certificate_dummy, runtime_raw, genre_dummy, directors_dummy, stars_dummy]
X1_raw = pd.concat(to_stack, axis = 1)
y1_raw = rating_raw

# Simple data splitting
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state = 14)



# _____________________ preparing the second dataset ________________________

dataset2 = dataset['Description']

corpus = []
for i in range(len(dataset2)):
    cleaned = prep.clean_text(dataset2[i], stemmer = 'Porter')
    corpus.append(cleaned)

# TF-idf vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X2 = tfidf.fit_transform(corpus).toarray()

# Simple data splitting
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y1, test_size = 0.2, random_state = 14)



# _________________________ first model ______________________________

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor # trees
from sklearn.svm import SVR # support vectors
from lightgbm import LGBMRegressor # boosting

# evaluate basic performance of the algorithms
regressors_1 = {'ExtraTrees': ExtraTreesRegressor(),
                'RandomForest': RandomForestRegressor(),
                'SVR': SVR(),
                'lightgbm': LGBMRegressor()}

cv = KFold(n_splits=10, shuffle=True, random_state=14)
scores = {'ExtraTrees': 0, 'RandomForest': 0, 'SVR': 0, 'lightgbm':0}
for model_name, model in regressors_1.items():
    n_score = cross_val_score(model, X1, y1, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    scores[model_name] = sc_rating.inverse_transform(mean(n_score))
    
# test RandomForestRegressor
cv = KFold(n_splits=5, shuffle=True, random_state=14)
model = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=14)
n_score = cross_val_score(model, X1, y1, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_score))

# RFR GridSearch
parameters_RFR1 = [{'n_estimators': [200, 300, 400],
                    'criterion': ['mae'],
                    'max_depth': [8, 10, 15, 20],
                    'min_samples_split': [1,2,3,4],
                    'n_jobs': [-1]}]
grid_search = GridSearchCV(estimator=model,
                           param_grid = parameters_RFR1,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = 3)
grid_search = grid_search.fit(X1, y1_raw)
print('Max RFR1 score = {}'.format(grid_search.best_score_))
best_RFR1_params = grid_search.best_params_

# test ExtraTreesRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
X_sint, y_sint = make_regression(n_samples=1000, n_features=10, n_informative=7, noise=0.1, random_state=3)
cv_sint = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
model = ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=3, random_state=14)
n_score_sint = cross_val_score(model, X_sint, y_sint, scoring='neg_mean_absolute_error', cv=cv_sint, n_jobs=-1, error_score='raise')
print(np.mean(n_score_sint))


cv = KFold(n_splits=5, shuffle=True, random_state=14)
model = ExtraTreesRegressor(n_estimators=100, max_depth=15, min_samples_split=5, max_features=8, n_jobs=-1, random_state=14)
n_score = cross_val_score(model, X1_raw, y1_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_score)) # -0.6 - not bad

# test RandomForestRegressor
cv = KFold(n_splits=5, shuffle=True, random_state=14)
model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=14)
n_score = cross_val_score(model, X1_raw, y1_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_score)) # -0.49 - better

# _________________________ second model - text processing ______________________________

# RandomForestRegressor
cv = KFold(n_splits=5, shuffle=True, random_state=14)
model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=14)
n_score = cross_val_score(model, X2, y1_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_score)) # -0.6 - 
# save the model
RFR2_filename = 'RFR_2set_0_6.sav'
joblib.dump(model, RFR2_filename)

# LGBMRegressor
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=14)
model = LGBMRegressor(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, bagging_fraction=0.7, n_estimators=300)
n_score = cross_val_score(model, X2, y1_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print(np.mean(n_score)) # -0.614 - 

# Applying GridSearch for LGBMRegressor

parameters = [{'num_leaves': [25, 28, 30, 32, 35], 
               'learning_rate': [0.02, 0.01, 0.005, 0.001], 
               'n_estimators': [350, 400, 500]}]
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           n_jobs = -1,
                           cv = 10)
grid_search = grid_search.fit(X2, y1_raw)
print(grid_search.best_score_)
best_params = grid_search.best_params_ # 0.593




