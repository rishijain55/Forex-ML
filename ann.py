# feed forward neural network with 2 hidden layers, input layer of 20 neurons, output layer of 1 neuron
#use tanh as activation function for hidden layers and linear for output layer

# Fitness_Function(chromosome)
# hidden_layer_1 := chromosome(0)
# hidden_layer_2 := chromosome(1)
# input_layer := 20
# output_layer := 1
# ann := CreateAnn(hidden_layer_1, hidden_layer_2, input_layer, output_layer)
# ann := TRAIN(ann)
# prediction := SIMULATE(ann)
# results := EVALUATE(prediction)
# wrong_predictions := GET_WRONG_PREDICTIONS(results)
# mae := mae(results)
# aof := a*mae + b*wrong_predictions
# return aof end

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
        self.model = MLPRegressor(hidden_layer_sizes=(self.hidden_layer_1, self.hidden_layer_2), activation=activation, solver=solver, max_iter=max_iter, random_state=random_state, validation_fraction=0.2, early_stopping=True, n_iter_no_change=10, verbose=True, nesterovs_momentum=True, learning_rate_init=0.01, learning_rate='adaptive')
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
    
    def get_wrong_predictions(self, X, y):
        X = self.scaler.transform(X)
        return np.sum(y != self.model.predict(X))
    
    def get_params(self):
        return self.model.get_params()
    
    def set_params(self, **params):
        return self.model.set_params(**params)
    
    

    
    


