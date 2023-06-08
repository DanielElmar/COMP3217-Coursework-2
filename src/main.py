import warnings

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def getDataComponents(data):

    pmu1 = data.iloc[:, :29]
    pmu2 = data.iloc[:, 29:58]
    pmu3 = data.iloc[:, 58:87]
    pmu4 = data.iloc[:, 87:116]
    logs = data.iloc[:, 116:len(data.columns)-1]
    labels = data.iloc[:, len(data.columns)-1]

    return pmu1, pmu2, pmu3, pmu4, logs, labels


def doLinearSVC(X, y, C=0.98):

    clf = LinearSVC(C=C)
    clf.fit(X,y)

    print("\n\nScores for Linear SVC\n")
    print(clf.score(X_test, y_test))

    #getScore(clf, X_train, y_train, X_test, y_test)

    return clf

def doKNeighbors(X, y, n_neighbors=1, leaf_size=1, p=1):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, p=p)
    knn.fit(X, y)

    print("\n\nScores for Linear SVC\n")
    print(knn.score(X_test, y_test))

    # getScore(clf, X_train, y_train, X_test, y_test)

    return knn

def doRandomForest(X,y, n_estimators=100, max_depth=10):
    
    rndf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rndf.fit(X, y)

    print("\n\nScores for Linear SVC\n")
    print(rndf.score(X_test, y_test))

    # getScore(clf, X_train, y_train, X_test, y_test)

    return rndf

def doLinearSVCGrid():

    # Grid Search
    params = [
        {
            "C": [i for i in np.arange(0, 1, 0.01)]
        }
    ]
    lin = GridSearchCV(LinearSVC(), param_grid=params, scoring="accuracy")
    lin.fit(X,y)

    # print best hyperparameters found
    print("Best hyperparameters")
    print(lin.best_params_)


def doKNeighbourGrid():

    # Grid Search
    params = [
        {
            "n_neighbors": [i for i in range(10)],
            "weights": ["uniform", "distance"],
            "algorithm": ["ball_tree", "kd_tree"],
            "leaf_size": [i for i in range(1, 2, 1)],
            "p": [i for i in range(1, 2)]
        }
    ]
    clf = GridSearchCV(KNeighborsClassifier(), param_grid=params, scoring="accuracy")
    clf.fit(X, y)

    # print best hyperparameters found
    print("Best hyperparameters")
    print(clf.best_params_)

def doRandomForestGrid():
    
    # Grid Search
    params = [
        {
            "n_estimators": [i for i in range(300, 600, 50)],
            "max_depth": [i for i in range(0, 500, 50)],
            "min_samples_split": [i for i in range(0, 10, 1)]
        }
    ]
    rndf = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring="accuracy")
    rndf.fit(X, y)

    # print best hyperparameters found
    print("Best hyperparameters")
    print(rndf.best_params_)


def doCrossValidation(clf, X, y, folds=5):

    # scoring to be recorded during the cross validation process
    scoring = ['accuracy']
    results = cross_validate(estimator=clf,
                             X=X,
                             y=y,
                             cv=folds,
                             scoring=scoring,
                             return_train_score=True,
                             return_estimator=True)

    # print the scores of the different folds
    print("\n\nScores for Cross Validation\n")

    print("train_accuracy: ", results['train_accuracy'])
    print("test_accuracy: ", results['test_accuracy'])

    # return 1 of the trained estimators
    return results["estimator"][4]


# more detailed score output if desired
def getScore(clf, X_train, y_train, X_test, y_test):

    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    print("score On Train: ", clf.score(X_train,y_train))
    print("Accuracy: ", accuracy_score(y_train, train_predict))
    print("F1: ", f1_score(y_train, train_predict))
    print("cross Val accuracy avg: ", cross_val_score(clf, X_train, y_train).mean())
    print("cross Val accuracy std: ", cross_val_score(clf, X_train, y_train).std())
    print("RMSE: ", math.sqrt(mean_squared_error(y_train, train_predict)))


    print("\nscore On Test: ", clf.score(X_test,y_test))
    print("Accuracy: ", accuracy_score(y_test, test_predict))
    print("F1: ", f1_score(y_test, test_predict))
    print("cross Val accuracy avg: ", cross_val_score(clf, X_test, y_test).mean())
    print("cross Val accuracy std: ", cross_val_score(clf, X_test, y_test).std())
    print("RMSE: ", math.sqrt(mean_squared_error(y_test, test_predict)))


# func to show relationship between the hyperparameter k and testing accuracy
def doGraphForKNeighbors():
    # try K=1 through K=25 and record testing accuracy
    k_range = range(1, 26)

    scores = []

    # loop through k values
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))


    # plot the relationship between K and testing accuracy
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')

    plt.show()

# fuc to remove columns containing only 1 value
def removeNonUnique(data):
    nunique = data.nunique()
    colsToDrop = nunique[nunique == 1].index
    data = data.drop(colsToDrop, axis=1)
    data.reset_index(drop=True)

    print("Cols Dropped: ", colsToDrop)

    return data


if __name__ == "__main__":

    partA = True


    if partA:
        trainingPath = "data/TrainingDataBinary.csv"
        testingPath = "data/TestingDataBinary.csv"
        outputPath = "TestingResultsBinary.csv"
    else:
        trainingPath = "data/TrainingDataMulti.csv"
        testingPath = "data/TestingDataMulti.csv"
        outputPath = "TestingResultsMulti.csv"


    # load the datasets using pandas
    data = pd.read_csv(trainingPath, header=None)
    unseen = pd.read_csv(testingPath, header=None)

    # Remove rows with NaN values
    data.dropna()

    # Shuffle Data
    data = shuffle(data)

    # Scale Data
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-1])
    unseen = scaler.fit_transform(unseen)

    # retrieve labels
    y = data.iloc[:, -1]

    # split training dataset into a new training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    print("Test: ", len(X_test))
    print("Train: ", len(X_train))




    # LINEAR CLASSIFIER

    # Func call to perform a grid search to find best hyperparameters for LinearSVC
    # doLinearSVCGrid()

    # Func call to perform a cross validation using a LinearSVC model for mode selection
    # lin = doCrossValidation(LinearSVC(), X_train, y_train)

    # Func call to train the LinearSVC and print results
    #model = doLinearSVC(X, y)









    # KNeighbors CLASSIFIER

    # Func call to perform a grid search to find best hyperparameters for KNeighborsClassifier
    # doKNeighbourGrid()

    # Func call to perform a cross validation using a KNeighborsClassifier model for mode selection
    # knn = doCrossValidation(KNeighborsClassifier(), X_train, y_train)

    # Func call to train the KNeighborsClassifier and print results
    #model = doKNeighbors(X, y)

    # func call to show graph for relationship between k and testing accuracy
    # doGraphForKNeighbors()










    # RANDOM FOREST CLASSIFIER

    # Func call to perform a grid search to find best hyperparameters for RandomForest
    # doRandomForestGrid()

    # Func call to perform a cross validation using a RandomForest model for mode selection
    # rndf = doCrossValidation(RandomForest(), X_train, y_train)

    # Func call to train the RandomForest and print results
    model = doRandomForest(X, y)


   

    # use the trained model to predict the labels for the testing data
    predicts = model.predict(unseen)
  
    # reload the testing dataset to retrieve un-scaled data
    unseen = pd.read_csv("data/TestingDataBinary.csv", header=None)

    # add labels column to end off the dataset
    unseen[128] = predicts

    # save testing dataset with predicted labels
    unseen.to_csv(outputPath, sep='\t', encoding='utf-8', index=False)


