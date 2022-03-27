import numpy as np

class Regression(object):
    """docstring for Regression."""
    def __init__(self, learning_rate=0.001, epochs=100):
        super(Regression, self).__init__()
        self.learning_rate = learning_rate
        self.epoch = epochs
        
    def learn_from_cost_function(self):
        """Applying Cost function to reduce the errors and give new values to Coef and intercepts"""
        """ Cost function = ((y- y_pred)**2)/no_of_rows"""

        self.y_pred = self.predict(self.X)

        derivate_w_r_t_coef_ = -2*((self.X.T).dot(self.y-self.y_pred))/(self.row)

        derivate_w_r_t_intercept_ = -2*(np.sum(self.y-self.y_pred))/(self.row)

        "By using Convergence theorem we will find new coef_ and intercepts"

        self.coef_ = self.coef_ - self.learning_rate*derivate_w_r_t_coef_

        self.intercept_ = self.intercept_ - self.learning_rate*derivate_w_r_t_intercept_

        return self
        
    """You can give any required inputs to the fit()"""
    def fit(self, X,y):
        """Here you can use the fit() from the LinearRegression of sklearn"""
        self.X = X
        self.y = y 
        self.row, self.col = X.shape
        self.coef_= np.zeros(self.col)
        self.intercept_ = 1
        for i in range(self.epoch):
            self.learn_from_cost_function()
        return f"Regression(learning_rate={self.learning_rate},epochs={self.epoch})"

    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""

    def predict(self,X_test):
        """ Write it from scratch using outcomes of fit()"""
        self.X_test = X_test
        y_hat = self.coef_.dot(self.X_test.T) + self.intercept_

        """Fill your code here. predict() should only take X_test and return predictions."""
        return y_hat

    def mean_squared_error(self,y_test, y_pred):
        """Here you can calculate the value of mean_squared_error"""
        self.y_test = y_test
        self_y_pred = y_pred

        return np.square(np.subtract(y_test, y_pred)).mean()

    def mean_absolute_error(self,y_test,y_pred):
        """Here you can calculate the value of mean_absolute_error."""
        self.y_test, self.y_pred = y_test, y_pred

        return abs(np.subtract(self.y_test,self_y_pred)).mean()
    
    def R2Square(self, y_test, y_pred):
        """Here, you can calculate the value of Rsquare."""
        self.y_test, self.y_pred = y_test, y_pred
        rss = np.square(np.subtract(y_test, y_pred)).sum()
        tss = np.square(y_test-y_test.mean()).sum()
        
        return 1-(rss/tss)
    
    def normalequations(self,X,y):
        """Using Normal equations."""
        self.X =X
        self.y =y
        self.coef_=np.linalg.inv((self.X.T).dot(self.X)).dot(self.X.T).dot(self.y)
        
        return self
    
    def predict_normal_equations(self, x_test):
        """Predictions on Normal Equations"""
        self.x_test = x_test 
        y_hat = self.coef_.dot(self.x_test.T)
        
        return y_hat