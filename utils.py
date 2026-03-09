import numpy as np
from sklearn.linear_model import LogisticRegression


__description__ = '''
Scikit-Learn no expone sus pesos matriciales fácilmente para ser enviados por red. 
Este archivo extrae e inyecta los parámetros matemáticos del modelo local.
'''

def get_model_parameters(model):
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]
    return params

def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model, n_features, n_classes):
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((1, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))