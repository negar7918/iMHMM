from iMHMM.tests import omhmm as oMHMM
from iMHMM.tests import mhmm as MHMM
from iMHMM.tests import imhmm as iMHMM
import numpy as np
import csv
from sklearn.metrics.cluster import v_measure_score
from scipy.linalg import eig
import math


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


input = np.empty((0, 91), dtype=object)
with open('./movement_libras.data', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        input = np.vstack((input, row))

input = input.astype(float)

data1 = np.empty((0, 90), dtype=float)
data2 = np.empty((0, 90), dtype=float)
i = 0
for label in input[:, -1]:
    if label == 4:
        data1 = np.vstack((data1, input[i, :-1]))
    elif label == 14:
        data2 = np.vstack((data2, input[i, :-1]))
    i += 1

data = np.concatenate((data1[:, :], data2[:, :]), axis=0)

np.random.seed(1)

num_of_cells = len(data[:, 0])
l = 90
y = np.zeros(num_of_cells, dtype=int)
M = 2
S = 2
lengths = (np.ones(num_of_cells, dtype=int) * l).tolist()

###################################################################################################

mhmm = oMHMM.SpaMHMM(epsilon=.0000001, n_nodes=1,
                       mix_dim=M,
                       n_components=S,
                       n_features=l,
                       graph=None,
                       n_iter=100,
                       verbose=True,
                       name='mhmm')
mhmm.fit(data.flatten()[:, np.newaxis], y, lengths)
pi_nk, transitions = mhmm._compute_mixture_posteriors(data.flatten()[:, np.newaxis], y, lengths)

trans = transitions.reshape(M, S, S)
print(trans)

print(np.exp(pi_nk))

predicted_cluster = []
label_0 = [0 for i in range(len(data1[:, 0]))]
label_1 = [1 for i in range(len(data2[:, 0]))]
for n in range(num_of_cells):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])

label = label_0 + label_1

print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('oMHMM v-measure: {}'.format(v))

print('entropy: {}'.format((calculate_entropy(trans[0]) + calculate_entropy(trans[1]))/2))
###################################################################################################

mhmm = MHMM.SpaMHMM(n_nodes=1,
                       mix_dim=M,
                       n_components=S,
                       n_features=l,
                       graph=None,
                       n_iter=100,
                       verbose=True,
                       name='mhmm')
mhmm.fit(data.flatten()[:, np.newaxis], y, lengths)
pi_nk, transitions = mhmm._compute_mixture_posteriors(data.flatten()[:, np.newaxis], y, lengths)

trans = transitions.reshape(M, S, S)
print(trans)

print(np.exp(pi_nk))

predicted_cluster = []
label_0 = [1 for i in range(len(data1[:, 0]))]
label_1 = [0 for i in range(len(data2[:, 0]))]
for n in range(num_of_cells):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])

label = label_0 + label_1

print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('MHMM v-measure: {}'.format(v))

print('entropy: {}'.format((calculate_entropy(trans[0]) + calculate_entropy(trans[1]))/2))

###################################################################################################

mhmm = iMHMM.SpaMHMM(length=l, n_nodes=1,
                       mix_dim=M,
                       n_components=S,
                       n_features=l,
                       graph=None,
                       n_iter=100,
                       verbose=True,
                       name='mhmm')
mhmm.fit(data.flatten()[:, np.newaxis], y, lengths)
pi_nk, transitions = mhmm._compute_mixture_posteriors(data.flatten()[:, np.newaxis], y, lengths)

trans = transitions.reshape(M, S, S)
print(trans)

print(np.exp(pi_nk))

predicted_cluster = []
label_0 = [0 for i in range(len(data1[:, 0]))]
label_1 = [1 for i in range(len(data2[:, 0]))]
for n in range(num_of_cells):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])

label = label_0 + label_1

print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('iMHMM v-measure: {}'.format(v))

print('entropy: {}'.format((calculate_entropy(trans[0]) + calculate_entropy(trans[1]))/2))