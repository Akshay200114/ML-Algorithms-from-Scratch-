import numpy as np
import pandas as pd

class LogisticRegression(object):
    """docstring for LogRegression."""
    def __init__(self, epochs, learning_rate):
        super(LogRegression, self).__init__()
        self.learning_rate = learning_rate 
        self.epoch = epochs

        
    def learn(self):
        """Update weights and intercept in this func"""
        self.y_pred = self.predict(self.X)
        
        d_coef_ = -2 * sum((self.y - self.y_pred).dot(self.y_pred).dot(1-self.y_pred).dot(self.X))
        
        d_intercept_ = -2* sum((self.y-self.y_pred).dot(self.y_pred).dot(1-self.y_pred))
        
        # By convergence Theorem
        self.coef_ = self.coef_ - self.learning_rate*d_coef_
        
        self.intercept_ = self.intercept_ - self.learning_rate*d_intercept_
        
        return self
        
    """You can give any required inputs to the fit()"""
    def fit(self, X, y):
        """Write it from scratch. Usage of sklearn is not allowed"""
        self.row, self.col = X.shape
        self.X = X
        self.y = y
        self.coef_ = np.zeros(self.col)
        self.intercept_ = 0
        for i in range(self.epoch):
            self.learn()
        return self
    

    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""

    def sigmoid(self, y):
        """Here, sigmoid function is used to range down  the values between 0 and 1"""
        return np.array([(1)/(1+np.exp(-(y)))])

    def predict(self,x_test):
        """Write it from scratch. Usage of sklearn is not allowed"""
        self.x_test = x_test
        self.y_hat = self.coef_.dot(self.x_test.T) + self.intercept_
        """Fill your code here. predict() should only take X_test and return predictions."""
        y_pred = self.sigmoid(self.y_hat)
        
        return y_predicted
