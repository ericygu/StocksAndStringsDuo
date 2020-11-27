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

def main():
    # Create xTrain, yTrain, xTest, yTest
    preprocessing.process()

    xTrain = file_to_numpy("xTrain")
    xTest = file_to_numpy("xTest")
    yTrain = file_to_numpy("yTrain")
    yTest = file_to_numpy("yTest")

    # K-fold cv
    kf = KFold(n_splits=10)
    xTrain, xTest, yTrain, yTest = kf.split(x, y)

    # Linear Regression (Closed)
    lr = LinearRegression.fit_transform(xTrain)
    yHat_lr = lr.predict(xTest)
    lr_testAcc = accuracy_score(yTest, yHat_lr)

    # Lasso Regression
    lasr = Lasso.fit_transform(xTrain)
    yHat_lasr = lasr.predict(xTest)
    lasr_testAcc = accuracy_score(yTest, yHat_lasr)

    # Ridge Regression
    ridr = Ridge.fit_transform(xTrain)
    yHat_ridr = ridr.predict(xTest)
    ridr_testAcc = accuracy_score(yTest, yHat_ridr)

    # ElasticNet
    elr = ElasticNet.fit_transform(xTrain)
    yHat_elr = elr.predict(xTest)
    elr_testAcc = accuracy_score(yTest, yHat_elr)

    # Print Statistics
    print("Model Accuracies")
    print("Linear Regression (Closed): ", lr_testAcc)
    print("Lasso Regression: ", lasr_testAcc)
    print("Ridge Regression: ", ridr_testAcc)
    print("Elastic Net: ", elr_testAcc)

if __name__ == '__main__':
    main()