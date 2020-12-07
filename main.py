import json
import re
from load_articles import read_articles, write_articles
from form_dictionary import read_dictionary
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
import preprocessing
import Math

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def graph():
    return None

def nested_cv(x, y, model, p_grid):
    # nested cv method can be condensed with the following code:
    """
    pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])
    cv = KFold(n_splits=4)
    scores = cross_val_score(pipeline, X, y, cv = cv)
    """
    nested_scores = list()

    # nested cv
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in outer_cv.split(x):
        # split data
        xTrain, xTest = x[train_index], x[test_index]
        yTrain, yTest = y[train_index], y[test_index]
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=1)
        xTrain, yTrain, xTest, yTest = preprocessing.process(xTrain, yTrain, xTest, yTest)

        # Scoring metric is roc_auc_score (precision and recall)
        clf = GridSearchCV(estimator=model, param_grid=p_grid, scoring='r2_score', cv=inner_cv, refit=True)
        eric_saves_the_day = clf.fit(xTrain, yTrain)
        best_model = eric_saves_the_day.best_estimator_
        yHat = best_model.predict(xTest)

        # Scoring
        r2 = r2_score(yTest, yHat)
        nested_scores.append(r2)
    return Math.mean(nested_scores), Math.std(nested_scores)

def kfold_cv(x, y, model):
    nested_scores = list()
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in outer_cv.split(x):
        # split data
        xTrain, xTest = x[train_index], x[test_index]
        yTrain, yTest = y[train_index], y[test_index]
        xTrain, yTrain, xTest, yTest = preprocessing.process(xTrain, yTrain, xTest, yTest)

        lr = model.fit(xTrain, yTrain)
        yHat = lr.predict(xTest)

        # Scoring
        r2 = r2_score(yTest, yHat)
        nested_scores.append(r2)
    return Math.mean(nested_scores), Math.std(nested_scores)

def main():
    # Retreive datasets
    # preprocessing.update_data()
    # preprocessing.get_csv()

    x = file_to_numpy("X.csv")
    y = file_to_numpy("Y.csv")

    # parameters being optimized, NEEDS EDITING
    """
    p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
    """
    p_grid = {}
    
    # Models and scores
    lr_r2_mean, lr_r2_std = kfold_cv(x,y,LinearRegression())
    lasso_r2_mean, lasso_r2_std = nested_cv(x,y,Lasso(),p_grid)
    ridge_r2_mean, ridge_r2_std = nested_cv(x,y,Ridge(),p_grid)
    enet_r2_mean, enet_r2_std = nested_cv(x,y,ElasticNet(),p_grid)

    """
    #------------------------------------------------
    # Models Original Setup (Old and can be removed)
    #------------------------------------------------
    # Linear Regression (Closed)
    lr = LinearRegression().fit(x_train, y_train)
    yHat_lr = lr.predict(x_train)
    lr_trainAcc = lr.score(x_train, y_train)
    yHat_lr = lr.predict(x_test)
    lr_testAcc = lr.score(x_test, y_test)

    # Lasso Regression
    lasr = Lasso().fit(x_train, y_train)
    yHat_lasr = lasr.predict(x_train)
    lasr_trainAcc = lasr.score(x_train, y_train)
    yHat_lasr = lasr.predict(x_test)
    lasr_testAcc = lasr.score(x_test, y_test)

    # Ridge Regression
    ridr = Ridge().fit(x_train, y_train)
    yHat_ridr = ridr.predict(x_train)
    ridr_trainAcc = ridr.score(x_train, y_train)
    yHat_ridr = ridr.predict(x_test)
    ridr_testAcc = ridr.score(x_test, y_test)

    # ElasticNet
    elr = ElasticNet().fit(x_train, y_train)
    yHat_elr = elr.predict(x_train)
    elr_trainAcc = elr.score(x_train, y_train)
    yHat_elr = elr.predict(x_test)
    elr_testAcc = elr.score(x_test, y_test)
    """

    # Print Statistics
    print("Model R^2 Scores")
    print("Linear Regression (Closed): [train]{} [test]{}".format(lr_trainAcc, lr_testAcc))
    print("Lasso Regression: [train]{} [test]{}".format(lasr_trainAcc, lasr_testAcc))
    print("Ridge Regression: [train]{} [test]{}".format(ridr_trainAcc, ridr_testAcc))
    print("Elastic Net: [train]{} [test]{}".format(elr_trainAcc, elr_testAcc))


if __name__ == '__main__':
    main()
