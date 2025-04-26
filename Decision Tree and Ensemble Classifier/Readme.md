## Problem Statement 
Predict whether a mushroom is edible or poisonous from a set of discrete attributes, namely cap-shape (6 possible values), cap-surface (4 possible values), cap-color (10 possible values),bruises (2 possible values), and odor (9 possible values). Data is given in the files as a comma separated list fe; x; s; y; t; ag where the first entry is the class (e or p), the second is the cap-shape (b, c, x, f, k, or s), the third is the cap-surface (f, g, y, or s), the fourth entry is the cap-color (n, b, c, g, r, p, u, e, w, y, t, or f), the fifth entry is whether it bruises (t, or f), and the last entry is the odor (a, l, c, y, f, m, n, p, or s).

## Data
There are two files. Training data is mu_train.csv in which last column has been eliminated.The column with odor has also been eliminated as it is a strong predictor of the resulatnt class.

It has been tested on mu_tst.csv.

## Decision Tree
The decision treee is constructed on basis of maximum information gain or minimum entropy at each level.

## Ensemble classifier:
I have made 2 files ensemble1.csv and ensemble2.csv based on random data points from mu_train.csv. It can be created automatically with program.
First it is trained on ensemble1.csv and checked for ensemble2.csv and then vice versa.
The accuracy of both is being printed by the program.
