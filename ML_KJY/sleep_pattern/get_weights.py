import numpy as np


weights = np.load('lweight.npy')
bias = np.load('lbias.npy')
lvari = np.load('lvari.npy')
v1 = lvari[0]
v2 = lvari[1]
v3 = lvari[2]
v4 = lvari[3]
v5 = lvari[4]

for i in lvari:
    print(i)


for i in weights:
    print(i)