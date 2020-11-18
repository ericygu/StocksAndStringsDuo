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

import preprocessing

def main():
   # Create xTrain, yTrain, xTest, yTest
   xTrain, yTrain, xTest, yTest = preprocessing.process()

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

   print("Model Accuracies")
   print("Linear Regression (Closed): ", lr_testAcc)
   print("Lasso Regression: ", lasr_testAcc)
   print("Ridge Regression: ", ridr_testAcc)
   print("Elastic Net: ", elr_testAcc)

if __name__ == '__main__':
    main()