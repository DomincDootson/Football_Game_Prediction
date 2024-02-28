from functools import partial

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier



def stepwise_selection(p, model, X_train, y_train):
    '''Uses backwards selection to select the p best features'''
    selector = RFE(model, n_features_to_select=1)
    selector = selector.fit(X_train, y_train)
    
    lst = [(r, c) for r, c in zip(selector.ranking_, X_train.columns)]
    lst.sort(key = lambda x : x[0])
    
    return [x[1] for x in lst[:p]], selector.estimator_

def stepwise_selection_with_error(p, model, metric, data_train, data_valid):
    ''' This function fits a linear model and selects the p best features
    Inputs
    ------

    p : int - The number of feature to select. 
    data_train : list[DataFrame, DataFrame] - the training data in the form [X_train, y_train]
    data_valid : list[DataFrame, DataFrame] - the validation data in the form [X_valid, y_valid]  

    Returns
    -------
    train_MSE : list[float] - MSE with the ith index the i+1th best feature evaluated on the training set
    valid_MSE : list[float] - MSE with the ith index the i+1th best feature evaluated on the validation set
    features : list[str] - The features in ranked order
    best_model - a list containing the best fit model (on the validation set), the features of this best fit and the score
    '''
    features, _  = stepwise_selection(p,model(), *data_train)

    train_MSE, valid_MSE = [], []
    min_val_MSE = -float('inf')
    for i,f in enumerate(features):
        new_model = model()
        fit_feat = features[:i+1]
        new_model.fit(data_train[0][fit_feat], data_train[1])
        
        train_MSE.append(metric(new_model.predict(data_train[0][fit_feat]), data_train[1]))
        valid_MSE.append(metric(new_model.predict(data_valid[0][fit_feat]), data_valid[1]))
        if valid_MSE[-1] > min_val_MSE:
            min_val_MSE = valid_MSE[-1]
            best_model = [new_model, fit_feat, new_model.score(data_valid[0][fit_feat], data_valid[1])]
            

    return train_MSE, valid_MSE, features, best_model

# def stepwise_selection_regression(p, data_train, data_valid, metric = mean_squared_error, model  = RidgeClassifier):
#     return stepwise_selection_with_error(p, data_train, data_valid, metric, model)

def stepwise_selection_classification(p, data_train, data_valid, metric = accuracy_score, model = RidgeClassifier, is_multilabel = False):
    if metric == f1_score:
        metric = partial(f1_score, average = 'weighted') # Means that we will deal with multi class, classification naivly
    if model is RidgeClassifier:
        model = partial(RidgeClassifier, alpha = 0) # This means we are doing classic linear classification

    if is_multilabel:
        model = partial(OneVsRestClassifier, estimator = model())


    return stepwise_selection_with_error(p, model,metric, data_train, data_valid)

def plotting_selection_result(t_mse, v_mse, ranked_features, best_model, valid_data):
    fig, axs = plt.subplots(ncols = 2, figsize=(15, 5))

    axs[0].plot(ranked_features, [t*0.75 for t in t_mse], label = 'Train MSE')
    axs[0].plot(ranked_features, v_mse, label = 'Validation MSE')
    axs[0].legend()
    axs[0].set_xticks(ranked_features, labels = [f"{f}, {i+1}" for i, f in enumerate(ranked_features)], rotation = 90)

    axs[1].hist(best_model[0].predict(valid_data[0][best_model[1]]), alpha = 1)
    axs[1].hist(valid_data[1], alpha = 0.8)
    plt.show()
    print(*enumerate(best_model[1]), sep = '\n')