import math
import numpy as np

import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    features = np.array(features)
    weights = np.array(weights)
    weights = weights.T

    ops = features @ weights + bias
    probabilities = [round(1 / (1 + math.exp(-op)), 4) for op in ops]
    
    mse = [math.pow((true - pred), 2) for true, pred in zip(labels, probabilities)]
    mse = round(sum(mse) / len(mse), 4)

    return probabilities, mse

single_neuron_model(features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1)