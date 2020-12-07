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
    yTrain = file_to_numpy("yTrain")
    yTest = file_to_numpy("yTest")

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
    lr = LinearRegression.fit_transform(x_train)
    yHat_lr = lr.predict(x_test)
    lr_testAcc = accuracy_score(y_test, yHat_lr)

    # Lasso Regression
    lasr = Lasso.fit_transform(x_train)
    yHat_lasr = lasr.predict(x_test)
    lasr_testAcc = accuracy_score(y_test, yHat_lasr)

    # Ridge Regression
    ridr = Ridge.fit_transform(x_train)
    yHat_ridr = ridr.predict(x_test)
    ridr_testAcc = accuracy_score(y_test, yHat_ridr)

    # ElasticNet
    elr = ElasticNet.fit_transform(x_train)
    yHat_elr = elr.predict(x_test)
    elr_testAcc = accuracy_score(y_test, yHat_elr)

    # Print Statistics
    print("Model Accuracies")
    print("Linear Regression (Closed): ", lr_testAcc)
    print("Lasso Regression: ", lasr_testAcc)
    print("Ridge Regression: ", ridr_testAcc)
    print("Elastic Net: ", elr_testAcc)


if __name__ == '__main__':
    main()
