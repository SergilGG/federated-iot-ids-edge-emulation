import numpy as np
from sklearn.linear_model import LogisticRegression


__description__ = '''
Scikit-Learn no expone sus pesos matriciales fácilmente para ser enviados por red. 
Este archivo extrae e inyecta los parámetros matemáticos del modelo local.
'''

def get_model_parameters(model):
    """
    Extraer los tensores internos del modelo (pesos y sesgo).
    Transformar los atributos de la Regresión Logística en una
    lista de matrices NumPy para poder enviarlos a travez de gRPC.
    """
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]
    return params

def set_model_params(model, params):
    """
    Inyectar los tensores globales promediados al recivir la
    actualización del servidor.
    Sobrescribir el estado local del modelo para inicializar
    la siguiente iteración de entrenamiento.
    """
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model, n_features, n_classes):
    """
    Construir la estructura inicial de tensores en ceros
    antes de la Ronda 1.
    Forzar la creación de los atributos coef_ e intercept_,
    los cuales Scikit-Learn no genera hasta ejecutar fit()
    por primera vez.
    """
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((1, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))