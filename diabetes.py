import os
import numpy
import pandas
import itertools

from sklearn import datasets, metrics, model_selection, preprocessing, linear_model, svm, naive_bayes, neighbors, neural_network, ensemble
from sklearn.externals import joblib
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataprovider import *


def DiabetesData():
    # Load Data
    dataframe = pandas.read_csv(config['diabetes']['dataset'])
    num_rows = dataframe.shape[0]
    print("%d Rows in Dataset" % num_rows)

    # Remove columns with missing elements
    dataframe = dataframe.dropna()
    num_rows = dataframe.shape[0]
    print("%d Rows in Dataset after removing null values" % num_rows)

    # Only predict 2 classes
    dataframe.loc[dataframe['Res'] != 0, 'Res'] = 1

    ## Create Feature Vectors
    features = dataframe.ix[:,:-1].values
    standard_scalar = preprocessing.StandardScaler().fit(features)
    features_std = standard_scalar.transform(features)

    predictions = dataframe.Res

    return (dataframe, standard_scalar, features_std, predictions)


def LogisticRegression(features, predictions):
    if(config['load_from_cache'] and os.path.isfile(config['diabetes']['LogisticRegression'])):
        return GetLogisticRegressionClassifierForDiabetes()

    classifier = linear_model.LogisticRegression(random_state=37)
    scores = model_selection.cross_val_score(classifier, features, predictions, cv=10, scoring='accuracy')
    print("\n%.2f%% accuracy with Logistic Regression" % (scores.mean() * 100))

    print("\nOptimizing Hyper-Parameters")
    c_values = numpy.arange(0.1,10,0.1).tolist()
    penalties = ['l1', 'l2']
    grid = model_selection.GridSearchCV(classifier, dict(C=c_values, penalty=penalties), cv=10, scoring='accuracy', n_jobs=-1, verbose=3 if config['debug'] else 0)
    grid.fit(features, predictions)

    print("%.2f%% accuracy with Logistic Regression (C = %f, Penalty = %s )" % (grid.best_score_ * 100, grid.best_params_['C'], grid.best_params_['penalty']))
    if config['persist']:
        StoreLogisticRegressionClassifierForDiabetes(grid.best_estimator_)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # i = 0
    # for param in grid.cv_results_['params']:
    #     penalty = 1 if param['penalty'] == 'l1' else 2;
    #     c = 'r' if penalty == 1 else 'b'
    #     ax.scatter(param['C'], penalty, grid.cv_results_['mean_test_score'][i]*100, c=c)
    #     i +=1
    #
    # ax.set_xlabel('C Value')
    # ax.set_ylabel('Penalty [ L1: 1, L2: 2]')
    # ax.set_zlabel('Accuracy')
    # plt.show()

    i=0
    for param in grid.cv_results_['params']:
        marker = 'o' if param['penalty'] == 'l1' else 's';
        color = 'r' if param['penalty'] == 'l1' else 'b';
        plt.plot(param['C'], grid.cv_results_['mean_test_score'][i]*100, c=color, marker=marker)
        i +=1

    plt.xlabel("C Value")
    plt.ylabel("Accuracy")
    plt.show()

    return grid.best_estimator_

def LinearSVC(features, predictions):
    if(config['load_from_cache'] and os.path.isfile(config['diabetes']['LinearSVC'])):
        return GetLinearSVCClassifierForDiabetes()

    classifier = svm.LinearSVC(random_state=37)
    scores = model_selection.cross_val_score(classifier, features, predictions, cv=10, scoring='accuracy')
    print("\n%.2f%% accuracy with Linear Support Vector Classification" % (scores.mean() * 100))

    print("\nOptimizing Hyper-Parameters")
    c_values = numpy.arange(0.1,25,0.1).tolist()
    grid = model_selection.GridSearchCV(classifier, dict(C=c_values), cv=10, scoring='accuracy', n_jobs=-1, verbose=3 if config['debug'] else 0)
    grid.fit(features, predictions)

    print("%.2f%% accuracy with Linear Support Vector Classification (C = %f )" % (grid.best_score_ * 100, grid.best_params_['C']))

    if config['persist']:
        StoreLinearSVCClassifierForDiabetes(grid.best_estimator_)

    # Plot graph of accuracy with c value
    plt.plot(c_values, grid.cv_results_['mean_test_score']*100)
    plt.xlabel("C Value")
    plt.ylabel("Accuracy")
    plt.show()

    return grid.best_estimator_


def NaiveBayes(features, predictions):
    if(config['load_from_cache'] and os.path.isfile(config['diabetes']['NaiveBayes'])):
        return GetNaiveBayesClassifierForDiabetes()

    classifier = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(classifier, features, predictions, cv=10, scoring='accuracy')
    print("\n%.2f%% accuracy with Gaussian Naive Bayes" % (scores.mean() * 100))

    grid = model_selection.GridSearchCV(classifier, dict(), cv=10, scoring='accuracy', n_jobs=-1, verbose=3 if config['debug'] else 0)
    grid.fit(features, predictions)

    if config['persist']:
        StoreNaiveBayesClassifierForDiabetes(grid.best_estimator_)

    return grid.best_estimator_


def KNeighbors(features, predictions):
    if(config['load_from_cache'] and os.path.isfile(config['diabetes']['KNeighbors'])):
        return GetKNeighborsClassifierForDiabetes()

    classifier = neighbors.KNeighborsClassifier()
    scores = model_selection.cross_val_score(classifier, features, predictions, cv=10, scoring='accuracy')
    print("\n%.2f%% accuracy with K Neighbors Classifier" % (scores.mean() * 100))

    print("\nOptimizing Hyper-Parameters")
    number_of_neighbors = numpy.arange(1,14).tolist()
    weights = ['uniform', 'distance']
    grid = model_selection.GridSearchCV(classifier, dict(n_neighbors=number_of_neighbors, weights=weights), cv=10, scoring='accuracy', n_jobs=-1, verbose=3 if config['debug'] else 0)
    grid.fit(features, predictions)

    print("%.2f%% accuracy with K Neighbors Classifier (n_neighbors = %d, weights = %s )" % (grid.best_score_ * 100, grid.best_params_['n_neighbors'], grid.best_params_['weights']))
    if config['persist']:
        StoreKNeighborsClassifierForDiabetes(grid.best_estimator_);

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # i = 0
    # for param in grid.cv_results_['params']:
    #     weight = 1 if param['weights'] == 'uniform' else 2;
    #     c = 'r' if weight == 1 else 'b'
    #     ax.scatter(param['n_neighbors'], weight, grid.cv_results_['mean_test_score'][i]*100, c=c)
    #     i +=1
    #
    # ax.set_xlabel('Number of Neighbors')
    # ax.set_ylabel('Weight [ Uniform: 1, Distance: 2]')
    # ax.set_zlabel('Accuracy')
    #
    # plt.show()

    i=0
    for param in grid.cv_results_['params']:
        marker = 'o' if param['weights'] == 'uniform' else 's';
        color = 'r' if param['weights'] == 'uniform' else 'b';
        plt.plot(param['n_neighbors'], grid.cv_results_['mean_test_score'][i]*100, c=color, marker=marker)
        i +=1
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()

    return grid.best_estimator_


def NN(features, predictions):
    if(config['load_from_cache'] and os.path.isfile(config['diabetes']['NeuralNetwork'])):
        return GetNeuralNetworkClassifierForDiabetes()

    classifier = neural_network.MLPClassifier(random_state=37,max_iter=500,alpha=1e-05,hidden_layer_sizes=(7,))
    scores = model_selection.cross_val_score(classifier, features, predictions, cv=10, scoring='accuracy')
    print("\n%.2f%% accuracy with MLP NN" % (scores.mean() * 100))

    print("\nOptimizing Hyper-Parameters")

    # hidden_layer_sizes = list(itertools.product(numpy.arange(5, 15, 1).tolist(), numpy.arange(5, 15, 1).tolist()))
    # activations= ['identity', 'logistic', 'tanh', 'relu']
    # solvers = ['lbfgs', 'sgd', 'adam']
    # learning_rate = ['constant', 'invscaling', 'adaptive']
    # alpha = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # hidden_layer_sizes=hidden_layer_sizes, activation=activations, solver=solvers, learning_rate=learning_rate, alpha=alpha
    # alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    grid = model_selection.GridSearchCV(classifier, {}, cv=10, scoring='accuracy', n_jobs=-1, verbose=3 if config['debug'] else 0)
    grid.fit(features, predictions)

    print("%.2f%% accuracy with MLP NN" % (grid.best_score_ * 100))
    if config['persist']:
        StoreNeuralNetworkClassifierForDiabetes(grid.best_estimator_)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # i = 0
    # for param in grid.cv_results_['params']:
    #     ax.scatter(param['hidden_layer_sizes'][0], param['hidden_layer_sizes'][1], grid.cv_results_['mean_test_score'][i]*100)
    #     i +=1
    #
    # ax.set_xlabel('Neurons in First Layer')
    # ax.set_ylabel('Neurons in Second Layer')
    # ax.set_zlabel('Accuracy')
    # plt.show()

    return grid.best_estimator_

def Ensemble(features, predictions):
    if(config['load_from_cache'] and os.path.isfile(config['diabetes']['Ensemble'])):
        return GetEnsembleClassifierForDiabetes()

    classifiers = list(zip(['lsvc', 'lr', 'nb', 'knn', 'nn'], GetAllClassifiersForDiabetes()))
    classifier = ensemble.VotingClassifier(estimators=classifiers)

    scores = model_selection.cross_val_score(classifier, features, predictions, cv=10, scoring='accuracy')
    print("\n%.2f%% accuracy with Ensemble Classifier" % (scores.mean() * 100))

    # print("\nOptimizing Hyper-Parameters")
    # params = { 'lr__C': numpy.arange(0.1,10,0.1).tolist(), 'lr__penalty': ['l1', 'l2'], 'lsvc__C': numpy.arange(0.1,10,0.1).tolist(), 'knn__n_neighbors': numpy.arange(1,14).tolist(), 'knn__weights': ['uniform', 'distance'] }
    grid = model_selection.GridSearchCV(classifier, {}, cv=10, scoring='accuracy', n_jobs=-1, verbose=3 if config['debug'] else 0)
    grid.fit(features, predictions)

    print("%.2f%% accuracy with Ensemble Voting Classifier" % (grid.best_score_ * 100))
    if config['persist']:
        StoreEnsembleClassifierForDiabetes(grid.best_estimator_);

    return grid.best_estimator_


def ShowScores(features, predictions_actual):
    classifiers = GetAllClassifiersForDiabetes() + (GetEnsembleClassifierForDiabetes(),)
    for classifier in classifiers:
        predicted = classifier.predict(features)
        print("\n--- %s ---" % (type(classifier)))
        print("| Accuracy Score: %.15f " % (metrics.accuracy_score(predictions_actual, predicted)))
        print("| Avg Precision Score: %.15f " % (metrics.average_precision_score(predictions_actual, predicted)))
        print("| F1 Score: %.15f " % (metrics.f1_score(predictions_actual, predicted)))
        print("| Log Loss: %.15f " % (metrics.log_loss(predictions_actual, predicted)))
        print("| Precision Score: %.15f " % (metrics.precision_score(predictions_actual, predicted)))
        print("| Recall Score: %.15f " % (metrics.recall_score(predictions_actual, predicted)))
        print("| AUC: %.15f " % (metrics.roc_auc_score(predictions_actual, predicted)))
        scoring = ['accuracy', 'average_precision', 'f1', 'neg_log_loss', 'precision', 'recall', 'roc_auc']
        for score in scoring:
            if (score == 'neg_log_loss' and type(classifier) is svm.LinearSVC)\
                or ((score == 'average_precision' or score == 'neg_log_loss' or score == 'roc_auc') and type(classifier) is ensemble.VotingClassifier):
                continue
            scores = model_selection.cross_val_score(classifier, features, predictions_actual, cv=10, scoring=score)
            print("| %s: %.15f " % (score, scores.mean()))
        print("--------------------------\n")


def main():
    dataframe, standard_scalar, features_std, predictions = DiabetesData()
    # StoreStandardScalarForDiabetes(standard_scalar)
    # logisticRegression = LogisticRegression(features_std, predictions)
    # linearsvc = LinearSVC(features_std, predictions)
    # naivebayes = NaiveBayes(features_std, predictions)
    # kneighbors = KNeighbors(features_std, predictions)
    # nn = NN(features_std, predictions)
    # ensemble = Ensemble(features_std, predictions)
    # ShowScores(features_std, predictions)

if __name__ == '__main__':
    main()
