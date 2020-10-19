import os
import time
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve, mean_squared_error
from scipy import stats

from sklearn.naive_bayes import MultinomialNB  # NB
from sklearn.neighbors import KNeighborsClassifier  # k-NN
from sklearn.linear_model import SGDClassifier, LinearRegression  # logistic regression
from sklearn.tree import DecisionTreeClassifier  # DT
from sklearn.svm import LinearSVC, SVR  # linear SVM\
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor

def pkl_write(data, filename='data.pickle'):
    with open(filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def save(best_model, filename):
    print("------SAVING MODEL------")
    pkl_write(best_model, filename + ".pickle")
    print("\t+ Done.")


def evaluate(grid_search, scaled_xtest, ytest):
    print("------EVALUATING MODEL-------")
    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(scaled_xtest)
    final_mse = mean_squared_error(ytest, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Root mean squared error on TEST: ", final_rmse)
    save(final_model, "RFR")

    #95% confidence interval 
    confidence = 0.95
    squared_errors = (final_predictions - ytest) ** 2
    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc = squared_errors.mean(), scale=stats.sem(squared_errors))))


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


def tune(forest_reg, scaled_xtrain, ytrain):
    param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},

    #then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(scaled_xtrain, ytrain)
    print("Best parameters: ", grid_search.best_params_)
    print("Best estimator: ", grid_search.best_estimator_)

    cv_res = grid_search.cv_results_
    for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
        print(np.sqrt(-mean_score), params)

    return grid_search


def scale(xtrain, ytrain, xtest, ytest):
    scaler = StandardScaler()
    scaler.fit(xtrain)
    scaled_xtrain = scaler.transform(xtrain)
    scaled_xtest = scaler.transform(xtest)
    scaled_xtrain = pd.DataFrame(scaled_xtrain, index=xtrain.index, columns=xtrain.columns)
    scaled_xtest = pd.DataFrame(scaled_xtest, index=xtest.index, columns=xtest.columns)
    return scaled_xtrain, scaled_xtest


def train():
    trainset = pd.read_csv("./data/train.csv")
    testset = pd.read_csv("./data/test.csv")
    xtrain, ytrain = trainset.drop(columns=['median_house_value']), trainset['median_house_value']
    xtest, ytest = testset.drop(columns=["median_house_value"]), testset["median_house_value"]

    print("-----TRAINING THE BASELINE-----")
    start = time.time()

    #scale data
    scaled_xtrain, scaled_xtest = scale(xtrain, ytrain, xtest, ytest)

    #train model
    forest_reg = RandomForestRegressor(n_estimators=5, random_state=5)
    forest_reg.fit(scaled_xtrain, ytrain)
    housing_predict = forest_reg.predict(scaled_xtrain)
    rfm_mse = mean_squared_error(ytrain, housing_predict)
    rfm_rmse = np.sqrt(rfm_mse)
    print(rfm_rmse)

    #compute scores
    rfm_scores = cross_val_score(forest_reg, scaled_xtrain, ytrain, scoring="neg_mean_squared_error", cv=5)
    frm_rmse_scores = np.sqrt(-rfm_scores)
    display_scores(frm_rmse_scores)

    #tune model
    grid_search = tune(forest_reg, scaled_xtrain, ytrain)

    #evaluate model
    evaluate(grid_search, scaled_xtest, ytest)
    end = time.time()


if __name__ == '__main__':
    train()


