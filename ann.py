import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


class ANN:
    def __init__(self, hidden_layer_1, hidden_layer_2, input_layer, output_layer, activation, solver, max_iter, random_state):
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.model = MLPRegressor(hidden_layer_sizes=(self.hidden_layer_1, self.hidden_layer_2), activation=activation, solver=solver, max_iter=max_iter, random_state=random_state, validation_fraction=0.2, early_stopping=True, n_iter_no_change=50, learning_rate_init=0.001)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_train, y_train = self.scaler.fit_transform(X), y
        self.model.fit(X_train, y_train)
        return self.model
    
    
    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        X = self.scaler.transform(X)
        return self.model.score(X, y)
    
    def mae(self, X, y):
        X = self.scaler.transform(X)
        return mean_squared_error(y, self.model.predict(X))
    
    def num_correct(self, y_test,y_pred):
        #correct is where sign of prediction and actual are the same
        return np.sum(np.sign(y_pred) == np.sign(y_test))
    
    def num_wrong(self, y_test,y_pred):
        #wrong is where sign of prediction and actual are different
        return np.sum(np.sign(y_pred) != np.sign(y_test))
    
    def get_params(self):
        return self.model.get_params()
    
    def set_params(self, **params):
        return self.model.set_params(**params)
    
    

    
    


