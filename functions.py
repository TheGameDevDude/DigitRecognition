import numpy as np

def relu(weighted_sum):
    return np.where(weighted_sum > 0, weighted_sum, 0)

def relu_prime(weighted_sum):
    return np.where(weighted_sum > 0, 1, 0)

def softmax(weighted_sum):
    if(weighted_sum.ndim == 1):
        exps = np.exp(weighted_sum - np.max(weighted_sum))
        return exps / np.sum(exps)
    else:
        normalized = weighted_sum - np.expand_dims(np.max(weighted_sum, axis=1), axis=1)
        exps = np.exp(normalized)
        return exps / np.expand_dims(np.sum(exps, axis=1), 1)

def softmax_prime(activations):
    return activations * (1 - activations)

def cross_entropy(output, target):
    return -(target * np.log(output)) - ((1-target) * np.log(1-output))

def cross_entropy_prime(output, target):
    return (output-target) / (output * (1-output))