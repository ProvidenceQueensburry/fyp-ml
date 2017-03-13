import os
import pickle
from sklearn.externals import joblib

config = {
    'heart': {
        'dataset': 'cleveland.csv',
        'scalar_file': 'models/heart/standard_scalar.pkl',
        'LinearSVC': 'models/heart/LinearSVC.pkl',
        'LogisticRegression': 'models/heart/LogisticRegression.pkl',
        'NaiveBayes': 'models/heart/NaiveBayes.pkl',
        'KNeighbors': 'models/heart/KNeighbors.pkl',
        'NeuralNetwork': 'models/heart/NN.pkl',
        'Ensemble' : 'models/heart/Ensemble.pkl'
    },
    'diabetes': {
        'dataset': 'pima-indians-diabetes.csv',
        'scalar_file': 'models/diabetes/standard_scalar.pkl',
        'LinearSVC': 'models/diabetes/LinearSVC.pkl',
        'LogisticRegression': 'models/diabetes/LogisticRegression.pkl',
        'NaiveBayes': 'models/diabetes/NaiveBayes.pkl',
        'KNeighbors': 'models/diabetes/KNeighbors.pkl',
        'NeuralNetwork': 'models/diabetes/NN.pkl',
        'Ensemble' : 'models/diabetes/Ensemble.pkl'
    },
    'persist': False,
    'load_from_cache': False,
    'debug': True
}

## Heart Disease
def StoreStandardScalarForHeart(standard_scalar):
    pickle.dump( standard_scalar, open( config['heart']['scalar_file'], 'wb' ) )

def GetStandardScalarForHeart():
    if os.path.isfile(config['heart']['scalar_file']):
        return pickle.load( open( config['heart']['scalar_file'], "rb" ) )
    return None

def GetAllClassifiersForHeart():
    return (GetLinearSVCClassifierForHeart(), GetLogisticRegressionClassifierForHeart(), GetNaiveBayesClassifierForHeart(), GetKNeighborsClassifierForHeart(), GetNeuralNetworkClassifierForHeart())

def GetLinearSVCClassifierForHeart():
    if os.path.isfile(config['heart']['LinearSVC']):
        return joblib.load(config['heart']['LinearSVC'])
    return None

def StoreLinearSVCClassifierForHeart(classifier):
    joblib.dump(classifier, (config['heart']['LinearSVC']));

def GetLogisticRegressionClassifierForHeart():
    if os.path.isfile(config['heart']['LogisticRegression']):
        return joblib.load(config['heart']['LogisticRegression'])
    return None

def StoreLogisticRegressionClassifierForHeart(classifier):
    joblib.dump(classifier, (config['heart']['LogisticRegression']));

def GetNaiveBayesClassifierForHeart():
    if os.path.isfile(config['heart']['NaiveBayes']):
        return joblib.load(config['heart']['NaiveBayes'])
    return None

def StoreNaiveBayesClassifierForHeart(classifier):
    joblib.dump(classifier, (config['heart']['NaiveBayes']));

def GetKNeighborsClassifierForHeart():
    if os.path.isfile(config['heart']['KNeighbors']):
        return joblib.load(config['heart']['KNeighbors'])
    return None

def StoreKNeighborsClassifierForHeart(classifier):
    joblib.dump(classifier, (config['heart']['KNeighbors']));

def GetNeuralNetworkClassifierForHeart():
    if os.path.isfile(config['heart']['NeuralNetwork']):
        return joblib.load(config['heart']['NeuralNetwork'])
    return None

def StoreNeuralNetworkClassifierForHeart(classifier):
    joblib.dump(classifier, (config['heart']['NeuralNetwork']));

def GetEnsembleClassifierForHeart():
    if os.path.isfile(config['heart']['Ensemble']):
        return joblib.load(config['heart']['Ensemble'])
    return None

def StoreEnsembleClassifierForHeart(classifier):
    joblib.dump(classifier, (config['heart']['Ensemble']));

## Diabetes

def GetAllClassifiersForDiabetes():
    return (GetLinearSVCClassifierForDiabetes(), GetLogisticRegressionClassifierForDiabetes(), GetNaiveBayesClassifierForDiabetes(), GetKNeighborsClassifierForDiabetes(), GetNeuralNetworkClassifierForDiabetes())

def StoreStandardScalarForDiabetes(standard_scalar):
    pickle.dump( standard_scalar, open( config['diabetes']['scalar_file'], 'wb' ) )

def GetStandardScalarForDiabetes():
    if os.path.isfile(config['diabetes']['scalar_file']):
        return pickle.load( open( config['diabetes']['scalar_file'], "rb" ) )
    return None

def GetLinearSVCClassifierForDiabetes():
    if(os.path.isfile(config['diabetes']['LinearSVC'])):
        return joblib.load(config['diabetes']['LinearSVC'])
    return None

def StoreLinearSVCClassifierForDiabetes(classifier):
    joblib.dump(classifier, (config['diabetes']['LinearSVC']));

def GetLogisticRegressionClassifierForDiabetes():
    if(os.path.isfile(config['diabetes']['LogisticRegression'])):
        return joblib.load(config['diabetes']['LogisticRegression'])
    return None

def StoreLogisticRegressionClassifierForDiabetes(classifier):
    joblib.dump(classifier, (config['diabetes']['LogisticRegression']));

def GetNaiveBayesClassifierForDiabetes():
    if(os.path.isfile(config['diabetes']['NaiveBayes'])):
        return joblib.load(config['diabetes']['NaiveBayes'])
    return None

def StoreNaiveBayesClassifierForDiabetes(classifier):
    joblib.dump(classifier, (config['diabetes']['NaiveBayes']));

def GetKNeighborsClassifierForDiabetes():
    if(os.path.isfile(config['diabetes']['KNeighbors'])):
        return joblib.load(config['diabetes']['KNeighbors'])
    return None

def StoreKNeighborsClassifierForDiabetes(classifier):
    joblib.dump(classifier, (config['diabetes']['KNeighbors']));

def GetNeuralNetworkClassifierForDiabetes():
    if(os.path.isfile(config['diabetes']['NeuralNetwork'])):
        return joblib.load(config['diabetes']['NeuralNetwork'])
    return None

def StoreNeuralNetworkClassifierForDiabetes(classifier):
    joblib.dump(classifier, (config['diabetes']['NeuralNetwork']));

def GetEnsembleClassifierForDiabetes():
    if(os.path.isfile(config['diabetes']['Ensemble'])):
        return joblib.load(config['diabetes']['Ensemble'])
    return None

def StoreEnsembleClassifierForDiabetes(classifier):
    joblib.dump(classifier, (config['diabetes']['Ensemble']));
