import numpy as np 

class Perceptron(object):
    def __init__(self,eta = 0.1, n_iter = 10):
        self.eta = eta 
        self.n_iter = n_iter

    def fit(self,X,y):
        #fit training data
        """
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.

        Returns
        ----------
        self: object
        """
        self.w = np.zeros(1+X.shape[1]) # Add w0
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self,X):
        #calculate net input
        return np.dot(X,self.w[1:])+self.w[0]

    def predict(self,X):
        # Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)

