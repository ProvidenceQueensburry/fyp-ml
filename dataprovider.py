import os
import pickle
from sklearn.externals import joblib

config = {
    'heart': {
        'scalar_file': 'models/heart/standard_scalar.pkl',
        'LinearSVC': 'models/heart/LinearSVC.pkl',
        'LogisticRegression': 'models/heart/LogisticRegression.pkl',
        'NaiveBayes': 'models/heart/NaiveBayes.pkl',
        'KNeighbors': 'models/heart/KNeighbors.pkl',
        'NeuralNetwork': 'models/heart/NN.pkl',
        'Ensemble' : 'models/heart/Ensemble.pkl'
    },
    'diabetes': {
        'scalar_file': 'models/diabetes/standard_scalar.pkl',
        'LinearSVC': 'models/diabetes/LinearSVC.pkl',
        'LogisticRegression': 'models/diabetes/LogisticRegression.pkl',
        'NaiveBayes': 'models/diabetes/NaiveBayes.pkl',
        'KNeighbors': 'models/diabetes/KNeighbors.pkl',
        'NeuralNetwork': 'models/diabetes/NN.pkl',
        'Ensemble' : 'models/diabetes/Ensemble.pkl'
    }
}


def GetJobLibFile(filepath):
    if os.path.isfile(filepath):
        return joblib.load(filepath)
    return None

def GetPickleFile(filepath):
    if os.path.isfile(filepath):
        return pickle.load( open(filepath, "rb" ) )
    return None

def GetStandardScalarForHeart():
    return GetPickleFile(config['heart']['scalar_file'])

def GetAllClassifiersForHeart():
    return (GetLinearSVCClassifierForHeart(), GetLogisticRegressionClassifierForHeart(), GetNaiveBayesClassifierForHeart(), GetKNeighborsClassifierForHeart(), GetNeuralNetworkClassifierForHeart(), GetEnsembleClassifierForHeart())

def GetLinearSVCClassifierForHeart():
    return GetJobLibFile(config['heart']['LinearSVC'])

def GetLogisticRegressionClassifierForHeart():
    return GetJobLibFile(config['heart']['LogisticRegression'])

def GetNaiveBayesClassifierForHeart():
    return GetJobLibFile(config['heart']['NaiveBayes'])

def GetKNeighborsClassifierForHeart():
    return GetJobLibFile(config['heart']['KNeighbors'])

def GetNeuralNetworkClassifierForHeart():
    return GetJobLibFile(config['heart']['NeuralNetwork'])

def GetEnsembleClassifierForHeart():
    return GetJobLibFile(config['heart']['Ensemble'])

## Diabetes

def GetAllClassifiersForDiabetes():
    return (GetLinearSVCClassifierForDiabetes(), GetLogisticRegressionClassifierForDiabetes(), GetNaiveBayesClassifierForDiabetes(), GetKNeighborsClassifierForDiabetes(), GetNeuralNetworkClassifierForDiabetes(), GetEnsembleClassifierForDiabetes())

def GetStandardScalarForDiabetes():
    return GetPickleFile(config['diabetes']['scalar_file'])

def GetLinearSVCClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LinearSVC'])

def GetLogisticRegressionClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LogisticRegression'])

def GetNaiveBayesClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['NaiveBayes'])

def GetKNeighborsClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['KNeighbors'])

def GetNeuralNetworkClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['NeuralNetwork'])

def GetEnsembleClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['Ensemble'])
