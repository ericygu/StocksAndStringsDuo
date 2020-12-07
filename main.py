import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
import preprocessing


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def df_to_numpy(xTrain, yTrain, xTest, yTest):
    return xTrain.to_numpy(), yTrain.to_numpy(), xTest.to_numpy(), yTest.to_numpy()


def graph():
    return None


def nested_cv(x, y, model, p_grid):
    # nested cv method can be condensed with the following code:
    """
    pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])
    cv = KFold(n_splits=4)
    scores = cross_val_score(pipeline, X, y, cv = cv)
    """
    nested_train_scores = list()
    nested_test_scores = list()

    # nested cv
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in outer_cv.split(x):
        # split data
        xTrain = x.iloc[train_index]
        xTest = x.iloc[test_index]
        yTrain = y.iloc[train_index]
        yTest = y.iloc[test_index]
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=1)
        xTrain, yTrain, xTest, yTest = preprocessing.process(xTrain, yTrain, xTest, yTest)

        xTrain, yTrain, xTest, yTest = df_to_numpy(xTrain, yTrain, xTest, yTest)

        # Scoring metric is roc_auc_score (precision and recall)
        clf = GridSearchCV(estimator=model, param_grid=p_grid, scoring='r2_score', cv=inner_cv, refit=True)
        fitter = clf.fit(xTrain, yTrain)
        best_model = fitter.best_estimator_

        # Train Score
        yHat1 = best_model.predict(xTrain)
        r2 = r2_score(yTrain, yHat1)
        nested_train_scores.append(r2)

        # Test Score
        yHat2 = best_model.predict(xTest)
        r2 = r2_score(yTest, yHat2)
        nested_test_scores.append(r2)
    return np.mean(nested_train_scores), np.std(nested_train_scores), np.mean(nested_test_scores), np.std(
        nested_test_scores)


def kfold_cv(x, y, model):
    nested_train_scores = list()
    nested_test_scores = list()

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, test_index in outer_cv.split(x):
        # split data
        xTrain = x.iloc[train_index]
        xTest = x.iloc[test_index]
        yTrain = y.iloc[train_index]
        yTest = y.iloc[test_index]
        xTrain, yTrain, xTest, yTest = preprocessing.process(xTrain, yTrain, xTest, yTest)

        xTrain, yTrain, xTest, yTest = df_to_numpy(xTrain, yTrain, xTest, yTest)

        md = model.fit(xTrain, yTrain)

        # Train Score
        yHat1 = md.predict(xTrain)
        r2 = r2_score(yTrain, yHat1)
        nested_train_scores.append(r2)

        # Test Score
        yHat2 = md.predict(xTest)
        r2 = r2_score(yTest, yHat2)
        nested_test_scores.append(r2)
    return np.mean(nested_train_scores), np.std(nested_train_scores), np.mean(nested_test_scores), np.std(
        nested_test_scores)


def main():
    # Retreive datasets
    # preprocessing.update_data()
    preprocessing.get_csv()

    x = pd.read_csv("X.csv")
    y = pd.read_csv("Y.csv")

    # parameters being optimized, NEEDS EDITING
    p_grid_lasso = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    p_grid_ridge = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    p_grid_enet = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

    # Models and scores
    print("starting")
    lr_trainScore, lr_trainStdev, lr_testScore, lr_testStdev = kfold_cv(x, y, LinearRegression())
    print("done with lr")
    lasr_trainScore, lasr_trainStdev, lasr_testScore, lasr_testStdev = nested_cv(x, y, Lasso(), p_grid_lasso)
    print("done with lasso")
    ridr_trainScore, ridr_trainStdev, ridr_testScore, ridr_testStdev = nested_cv(x, y, Ridge(), p_grid_ridge)
    print("done with ridge")
    elr_trainScore, elr_trainStdev, elr_testScore, elr_testStdev = nested_cv(x, y, ElasticNet(), p_grid_enet)
    print("done with elastic net")

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
    print("Linear Regression (Closed): [train]{} [test]{}".format(lr_trainScore, lr_trainStdev, lr_testScore,
                                                                  lr_testStdev))
    print(
        "Lasso Regression: [train]{} [test]{}".format(lasr_trainScore, lasr_trainStdev, lasr_testScore, lasr_testStdev))
    print(
        "Ridge Regression: [train]{} [test]{}".format(ridr_trainScore, ridr_trainStdev, ridr_testScore, ridr_testStdev))
    print("Elastic Net: [train]{} [test]{}".format(elr_trainScore, elr_trainStdev, elr_testScore, elr_testStdev))


if __name__ == '__main__':
    main()
