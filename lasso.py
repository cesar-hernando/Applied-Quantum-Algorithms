'''
In this file, I implement the LASSO model to learn the coefficients that characterize the unknown observable
'''

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def lasso_regression(X, y):
    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define an object of the class Lasso and train the model
    lasso = LassoCV(alphas=[1e-4,1e-3,1e-2,1e-1,1], cv=5, random_state=42)
    lasso.fit(X_train, y_train)
    optimal_alpha = lasso.alpha_
    coefficients = lasso.coef_
    L1_norm_coef = np.linalg.norm(coefficients, ord=1)

    # Calculate the predictions for the test set
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)

    # Evaluate the error and R^2 score in training data and test
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    return coefficients, L1_norm_coef, optimal_alpha, train_mse, test_mse, train_r2, test_r2


   
    



