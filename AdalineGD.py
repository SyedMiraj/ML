import numpy as np

# =============================================================================
# Adative Linear Neurone CLassfier
# =============================================================================
class AdalineGD (object):
# =============================================================================
#     parameter
#     ---------
#     eta -> learning rate (ranges from 0.0 to 1.0)
#     n_iter -> passes over the training dataset
#    
#    attribute
#    ---------
#    w_ : 1d array [weights after fitting]
#    errors_ : list [number of misclassification in every epoch]
# =============================================================================
    
    def __init__(self, eta = .01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
      
# =============================================================================
#         Method for fitting data
# =============================================================================
    def fit(self, X, y): 
# =============================================================================
#     Parameters        
#     ---------        
#     X : {array-like}, shape = [n_samples, n_features]            
#     Training vectors,             
#     where n_samples is the number of samples and            
#     n_features is the number of features.        
#     
#     y : array-like, shape = [n_samples]            
#     Target values
#     
#      Returns        
#      ------        
#      self : object
# =============================================================================
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2) .sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.w_[1:], self.w_[0])
    
    def activation(self, X):
        """ Compute linear activation """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """ Return unit label after unit step """
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
        