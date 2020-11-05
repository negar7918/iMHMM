import numpy as np
import csv
from sklearn.metrics.cluster import v_measure_score
from iMHMM.tests import imhmm
from scipy.linalg import eig
import math
import matplotlib.pylab as plt


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


def plot(c, name, color):
    fig, ax = plt.subplots(1)
    plt.plot(np.arange(904), c, color=color)
    plt.xlabel('sequence position')
    plt.ylabel('value')
    ax.set_ylim([0, 1000])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('movement ' + name)
    plt.show()
    fig.savefig('./cluster' + name + '.png')


input = np.empty((0,37), dtype=object)
with open('./chrom_18_to_21.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        input = np.vstack((input, row))


Xtrain = np.swapaxes(input[1:, 1:], 0, 1).astype(int)

data = Xtrain[:, :]

np.random.seed(1)

l = 903
y = np.zeros(36, dtype=int)
M = 2
S = 6
lengths = (np.ones(36, dtype=int) * l).tolist()
mhmm = imhmm.SpaMHMM(length=l, n_nodes=1,
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
label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for n in range(36):
    cell = np.float64(pi_nk[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell))[0][0])
    if label[n] == 0:
        label[n] = 1
    else:
        label[n] = 0

plot(Xtrain[1, :], 'metastatic', 'r')
plot(Xtrain[33, :], 'primary', 'b')


print(label)
print(predicted_cluster)

v = v_measure_score(label, predicted_cluster)
print('v-measure: {}'.format(v))

print('entropy: {}'.format((calculate_entropy(trans[0]) + calculate_entropy(trans[1]))/2))

