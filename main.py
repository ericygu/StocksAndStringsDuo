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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold
import preprocessing


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def graph():
    return


def main():
    # Create xTrain, yTrain, xTest, yTest
    # preprocessing.process()

    xTrain = file_to_numpy("xTrain_pearson.csv")
    xTest = file_to_numpy("xTest_pearson.csv")
    yTrain = file_to_numpy("yTrain.csv")
    yTest = file_to_numpy("yTest.csv")

    # temp
    x_train = xTrain
    y_train = yTrain
    x_test = xTest
    y_test = yTest

    # K-fold cv
    """
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(xTrain):
        x_train, x_test = xTrain[train_index], xTrain[test_index]
        y_train, y_test = yTrain[train_index], yTest[test_index]
    """
    
    # cross_val_score
    # GridSearchCV

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

    # Print Statistics
    print("Model R^2 Scores")
    print("Linear Regression (Closed): [train]{} [test]{}".format(lr_trainAcc, lr_testAcc))
    print("Lasso Regression: [train]{} [test]{}".format(lasr_trainAcc, lasr_testAcc))
    print("Ridge Regression: [train]{} [test]{}".format(ridr_trainAcc, ridr_testAcc))
    print("Elastic Net: [train]{} [test]{}".format(elr_trainAcc, elr_testAcc))


if __name__ == '__main__':
    main()
