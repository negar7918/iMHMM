from scipy.linalg import eig
import math
import numpy as np


def calculate_entropy(transition):
    lam, vec = eig(transition, left=True, right=False)
    idx = np.argmin(np.abs(lam - 1))
    w = np.real(vec[:, idx])
    mu = w / w.sum()
    num_of_states = len(transition[0])
    entropy = 0
    for i in range(num_of_states):
        entropy += mu[i] * np.sum([p * math.log(p + .01) for p in transition[i]])

    return -entropy

A = np.array([[.99, .01], [.02, .98]])
B = np.array([[.4, .6], [.5, .5]])

N = np.array([[.99, 0, .01], [.1, .8, .1], [.01, 0, .99]])
M = np.array([[.5, .2, .3], [.4, .3, .3], [.2, .4, .4]])

print(calculate_entropy(N))
print(calculate_entropy(M))
print((calculate_entropy(N) + calculate_entropy(M))/2)

#print((calculate_entropy(A) + calculate_entropy(B))/2)