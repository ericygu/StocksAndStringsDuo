from statistics import mean, stdev
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
import preprocessing
from tqdm import tqdm


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
    nested_params = list()

    # nested cv
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in tqdm(outer_cv.split(x)):
        # split data
        xTrain = x.iloc[train_index]
        xTest = x.iloc[test_index]
        yTrain = y.iloc[train_index]
        yTest = y.iloc[test_index]
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
        xTrain, yTrain, xTest, yTest = preprocessing.process(xTrain, yTrain, xTest, yTest)

        # PCA (number of components chosen such that the amount of variance 
        # that needs to be explained is greater than the percentage specified by n_components)
        sklearn_PCA = PCA(n_components=0.95, svd_solver='full')
        xTrain = sklearn_PCA.fit_transform(xTrain)
        xTest = sklearn_PCA.transform(xTest)
        # xTrain, yTrain, xTest, yTest = df_to_numpy(xTrain, yTrain, xTest, yTest)

        # Scoring metric is roc_auc_score (precision and recall)
        clf = GridSearchCV(estimator=model, param_grid=p_grid, scoring='r2', cv=inner_cv, refit=True)
        fitter = clf.fit(xTrain, yTrain)
        best_model = fitter.best_estimator_
        nested_params.append(fitter.best_params_)

        # Train Score
        yHat1 = best_model.predict(xTrain)
        r2 = r2_score(yTrain, yHat1)
        nested_train_scores.append(r2)

        # Test Score
        yHat2 = best_model.predict(xTest)
        r2 = r2_score(yTest, yHat2)
        nested_test_scores.append(r2)
    return mean(nested_train_scores), stdev(nested_train_scores), mean(nested_test_scores), stdev(nested_test_scores),nested_train_scores, nested_test_scores, nested_params


def kfold_cv(x, y, model):
    nested_train_scores = list()
    nested_test_scores = list()

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in tqdm(outer_cv.split(x)):
        # split data
        xTrain = x.iloc[train_index]
        xTest = x.iloc[test_index]
        yTrain = y.iloc[train_index]
        yTest = y.iloc[test_index]
        xTrain, yTrain, xTest, yTest = preprocessing.process(xTrain, yTrain, xTest, yTest)

        # PCA (number of components chosen such that the amount of variance 
        # that needs to be explained is greater than the percentage specified by n_components)
        sklearn_PCA = PCA(n_components=0.95, svd_solver='full')
        xTrain = sklearn_PCA.fit_transform(xTrain)
        xTest = sklearn_PCA.transform(xTest)
        # xTrain, yTrain, xTest, yTest = df_to_numpy(xTrain, yTrain, xTest, yTest)

        md = model.fit(xTrain, yTrain)

        # Train Score
        yHat1 = md.predict(xTrain)
        r2 = r2_score(yTrain, yHat1)
        nested_train_scores.append(r2)

        # Test Score
        yHat2 = md.predict(xTest)
        r2 = r2_score(yTest, yHat2)
        nested_test_scores.append(r2)
    return nested_train_scores, nested_test_scores


def main():
    # Retrieve datasets
    # preprocessing.update_data()
    x, y = preprocessing.get_csv()

    # parameters being optimized, ranges were determined from self-testing.
    p_grid_lasso = {"alpha": [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 1, 5]}
    p_grid_ridge = {"alpha": [0.1, 0.5, 1.0, 5, 10, 20, 30, 40 , 50, 60]}
    p_grid_enet = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

    # Models and scores
    print("starting")
    lrTrainScores, lrTestScores = kfold_cv(x, y, LinearRegression())

    print("Lineaer Regression R^2 Scores")
    print("Training Mean:", mean(lrTrainScores), "+/-", stdev(lrTrainScores))
    print("Testing Mean:", mean(lrTestScores), "+/-", stdev(lrTestScores))

    laTrainScores, laTestScores, laParams = nested_cv(x, y, Lasso(), p_grid_lasso)

    print("Lasso Regression R^2 Scores")
    print("Training Mean:", mean(laTrainScores), "+/-", stdev(laTrainScores))
    print("Testing Mean:", mean(laTestScores), "+/-", stdev(laTestScores))
    print("Best Parameters:", laParams)
    print("Individual Train Scores:", laTrainScores)
    print("Individual Test Scores:", laTestScores)

    riTrainScores, riTestScores, riParams = nested_cv(x, y, Ridge(), p_grid_ridge)

    print("Ridge Regression R^2 Scores")
    print("Training Mean:", mean(riTrainScores), "+/-", stdev(riTrainScores))
    print("Testing Mean:", mean(riTestScores), "+/-", stdev(riTestScores))
    print("Best Parameters:", riParams)
    print("Individual Train Scores:", riTrainScores)
    print("Individual Test Scores:", riTestScores)

    elTrainScores,elTestScores, elParams = nested_cv(x, y, ElasticNet(), p_grid_enet)

    print("Elastic Net R^2 Scores")
    print("Training Mean:", mean(elTrainScores), "+/-", stdev(elTrainScores))
    print("Testing Mean:", mean(elTestScores), "+/-", stdev(elTestScores))
    print("Best Parameters:", elParams)
    print("Individual Train Scores:", elTrainScores)
    print("Individual Test Scores:", elTestScores)

if __name__ == '__main__':
    main()
