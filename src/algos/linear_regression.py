import numpy as np
from gradient_descent import gradient_descent

class LinearRegressionAnalytical:
    def __init__(self):
        self._weights = None

    def fit(self, X, y):
        # Place a column of ones as a dummy input attribute to represent linear function as a dot product of weights and input vectors 
        X = np.insert(X, 0, np.ones_like(int, shape=(X.size)), axis=1)

        X_transpose = np.transpose(X)
        X_pseudo_inverse = np.linalg.inv(np.dot(X_transpose, X))

        self._weights = np.dot(np.dot(X_pseudo_inverse, X_transpose), y)
        return self

    def predict(self, X):
        X = np.insert(X, 0, np.ones_like(int, shape=(X.size)), axis=1)

        # Transpose X so we can get along with the np.dot()
        X = np.transpose(X)

        return np.dot(self._weights, X)
    

class LinearRegressionGradientDescent:
    def __init__(self):
        self._weights = None

    def fit(self, X, y):
        # Define partial_derivate factory
        def make_partial_derivate(i : int):
            def partial_derivate(weights : np.ndarray):
                sum = 0
                for input, output in zip(X, y):
                    estimate = np.dot(weights, input)
                    sum += (output - estimate) * input[i]
                return sum
            return partial_derivate
        
        # Define converged factory
        def make_converged():
            THRESHOLD = 0.01
            closure : list[np.ndarray, int] = [[], 0]
    
            def converged(weight : np.ndarray):
                last_weight = closure[0]
                if len(last_weight) == 0:
                    closure[0] = weight.copy()
                    return False
    
                distance = np.linalg.norm(weight - last_weight)

                below_threshold_iterations = closure[1]
            
                if distance < THRESHOLD:
                    below_threshold_iterations += 1
                else:
                    below_threshold_iterations = 0 

                closure[0] = weight.copy()
                closure[1] = below_threshold_iterations
                if below_threshold_iterations > 5:
                    return True               
                
            return converged
        
        # Place a column of ones as a dummy input attribute to represent linear function as a dot product of weights and input vectors 
        X = np.insert(X, 0, np.ones_like(int, shape=(X.size)), axis=1)
        dimension = X.shape[1]

        start = np.array([0, 0])
        partial_derivatives = [make_partial_derivate(i) for i in range(dimension)]
        converged = make_converged()            

        self._weights = gradient_descent(start, 0.5, partial_derivatives, converged)
        return self

    def predict(self, X):
        X = np.insert(X, 0, np.ones_like(int, shape=(X.size)), axis=1)

        # Transpose X so we can get along with the np.dot()
        X = np.transpose(X)

        return np.dot(self._weights, X)
    