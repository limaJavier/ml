import numpy as np


class NaiveBayesClassifier:
    def __init__(self, weight : float, domains : list[int]):
        effects = domains[1:]
        
        self._weight = weight
        self._causes = [0 for _ in range(domains[0])]
        self._effects : dict[int, list[list[float]]] = {}

        for i, dimension in enumerate(effects):
            self._effects[i] = []
            for _ in self._causes:
                self._effects[i].append([0 for _ in range(dimension)])

    def fit(self, X : list[list[int]], y : list[int]):        
        for input, cause in zip(X, y):
            self._causes[cause] += 1
            for effect, value in enumerate(input):
                self._effects[effect][cause][value] += 1

        for effect, conditional_matrix in self._effects.items():
            for cause, conditional_row in enumerate(conditional_matrix):
                zero_count = 0
                for conditional_probability in conditional_row:
                    if conditional_probability == 0:
                        zero_count += 1
                zero_increment = self._weight / zero_count if zero_count > 0 else 0

                for value, effect_count in enumerate(conditional_row):
                    if effect_count == 0:
                        self._effects[effect][cause][value] += zero_increment
                    else:
                        ratio = effect_count / self._causes[cause]
                        self._effects[effect][cause][value] = ratio * (1 - self._weight) if zero_increment > 0 else ratio
    
        return self
    
    def predict(self, X):
        y = []
        for input in X:
            log_sums = []
            for cause, _ in enumerate(self._causes):
                cause_probability = self._causes[cause] / len(self._causes)
                log_sum = np.log(cause_probability)
                for effect, value in enumerate(input):
                    conditional_probability = self._effects[effect][cause][value]
                    if conditional_probability == 0:
                        continue
                    log_sum += np.log(conditional_probability)               
                log_sums.append(log_sum)

            most_probable_cause : int = None
            greatest_sum = -pow(2, 32)
            for cause, sum in enumerate(log_sums):
                if sum > greatest_sum:
                    greatest_sum = sum
                    most_probable_cause = cause
            
            y.append(most_probable_cause)

        return y
                    
                    