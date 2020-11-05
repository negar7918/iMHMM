# Informational Mixture of Hidden Markov Models #


The iMHMM implementation is based on the standard EM for MHMM from this paper:

 Pernes, D.; and Cardoso, J. S.
 Spamhmm: Sparse mixture of hidden markov models for graph connected entities.
 2019 International Joint Conference on Neural Networks(IJCNN)
 pp. 1–10
 (2019)

and also oMHMM from this paper:

 Safinianaini, N.; de Souza, C.; Bostr̈om, H.; and Lagergren,J.
 Orthogonal mixture of hidden markov models
 (ECML-PKDD)
 (2020)

---------------------------------------------

The softwares needed to run the experiments:

Python 3.6.2
hmmlearn 0.2.1
cvxpy 1.0.21
numpy 1.16.2
scikit-learn 0.19.1
scipy 1.1.0

---------------------------------------------

The computing infrastructure which we use:

OS: OS X

Processor: 2,8 GHz Intel Core i7

Memory: 16 GB 2133 MHz LPDDR3

Graphics: Radeon Pro 560 4 GB
          Intel HD Graphics 630 1536 MB

---------------------------------------------

Note: To see the regularization introduced by iMHMM, see line 59 of the file "imhmm.py" in the test folder.
