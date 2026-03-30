import numpy as np 
import math 

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy(y_logits):
    y_logits_softmax = softmax(y_logits)
    probs = y_logits_softmax[np.arange(len(y_true)), y_true]
    log_prob = -np.log(probs)
    return np.mean(log_prob)

y_logits=np.array([
    [0.0, 5.0, 0.0, 0.0],
    [1.0, 0.0, 0.1, 5.0],
    [0.0, 1.0, 5.0, 1.0],
    [0.0, 1.0, 1.0, 5.0]
])
y_true = np.array([0,1,2,3])
print(cross_entropy(y_logits))
