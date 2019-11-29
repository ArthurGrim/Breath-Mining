import numpy as np
import sklearn
import math
import itertools
import pandas as pd
import os
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from munkres import Munkres
from scipy.spatial import distance
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

peakAlignement = []

def getRawDataAslist(filename):
    print(filename)
    rawMat = pd.read_csv(filename,low_memory=False, sep=",", comment='#')
    rawMat = rawMat.drop(rawMat.index[[0]])
    rawMat = rawMat.drop(rawMat.columns[[0,1,2]], axis=1)
    rawMat = rawMat.to_numpy()
    rawList = rawMat.flatten()
    return rawList.tolist()

def getRawDataMatrix(basepath):
    mat = []
    fileNames = []
    for entry in sorted(os.listdir(basepath)):
        if os.path.isfile(os.path.join(basepath, entry)):
            if "lock" not in entry:
                mat.append(getRawDataAslist(basepath+entry))
                fileNames.append(entry)
    return mat,fileNames

def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]




M,fileNames = getRawDataMatrix("Data/candy_raw/")

linked = linkage(M, 'single')

t = to_tree(linked)
print(t.pre_order())

alignementOrder = np.delete(np.array(linked),np.s_[2,3],1).astype(int)
print(alignementOrder)

# plt.figure(figsize=(10, 7))
# dendrogram(linked, labels=range(0,6))
# plt.show()





nodesAlignements = {}
distanceThreshold = 2 #minimal distance for pairing 2 points
alignementIndex = len(fileNames)-1

for p in range(len(fileNames)):
    pl = pd.read_csv("Data/peak_identification/"+fileNames[p],low_memory=False, sep="\t", comment='#')[['t','r']]
    pl["t"] *= 100

    pl = pl.to_numpy().tolist()
    print(pl)
    nodesAlignements[p] = pl


m = Munkres()

for pairs in alignementOrder: #for each node in the tree
    print(len(nodesAlignements[pairs[0]]))
    print(len(nodesAlignements[pairs[1]]))
    if len(nodesAlignements[pairs[0]]) > len(nodesAlignements[pairs[1]]):
        pairs = [pairs[1],pairs[0]]
    print(pairs)
    alignementIndex+=1
    nodesAlignements[alignementIndex] = []
    print(alignementIndex)
    distMat = distance_matrix(nodesAlignements[pairs[0]], nodesAlignements[pairs[1]])



##probleme dans l'ordre des axes x y dans la matrice de distance
    indexes = m.compute(distMat)
    total = 0
    for row, column in indexes:
        dist = distance.euclidean(nodesAlignements[pairs[0]][row],nodesAlignements[pairs[1]][column])

        if dist<distanceThreshold:
            print("Paired")
            midpointCoordinates = midpoint(nodesAlignements[pairs[0]][row],nodesAlignements[pairs[1]][column])
            nodesAlignements[alignementIndex].append(midpointCoordinates)
        else:
            print("Unique")
            nodesAlignements[alignementIndex].append(nodesAlignements[pairs[0]][row])
            nodesAlignements[alignementIndex].append(nodesAlignements[pairs[1]][column])


alignementEx = pd.read_csv("Data/peakAlignement.txt",low_memory=False, sep=";")
alignementEx["t"] *= 100
alignementEx["r"] *= 500

print(alignementEx)

# sns.heatmap(distMat)
# plt.show()

plt.scatter(np.array(nodesAlignements[0]).transpose()[0], np.array(nodesAlignements[0]).transpose()[1], c="purple")
plt.scatter(np.array(nodesAlignements[1]).transpose()[0], np.array(nodesAlignements[1]).transpose()[1], c="blue")
plt.scatter(np.array(nodesAlignements[2]).transpose()[0], np.array(nodesAlignements[2]).transpose()[1], c="red")
plt.scatter(np.array(nodesAlignements[3]).transpose()[0], np.array(nodesAlignements[3]).transpose()[1], c="orange")
plt.scatter(np.array(nodesAlignements[4]).transpose()[0], np.array(nodesAlignements[4]).transpose()[1], c="yellow")
plt.scatter(np.array(nodesAlignements[5]).transpose()[0], np.array(nodesAlignements[5]).transpose()[1], c="brown")


# plt.scatter(alignementEx["t"], alignementEx["r"], c="red",marker = "1" )
plt.scatter(np.array(nodesAlignements[10]).transpose()[0], np.array(nodesAlignements[10]).transpose()[1], c="green",marker = "2" )
plt.show()






print(nodesAlignements[10])

print(len(nodesAlignements[10]))








# def getLinkageConstrain(df):
#
#     cannot_link = []
#     group = []
#     grpName = df["measurement_name"].iloc[0]
#     print(grpName)
#     for index, row in df.iterrows():
#         if grpName == row["measurement_name"]:
#             group.append(index)
#         else:
#             grpName = row["measurement_name"]
#             group = list(itertools.combinations(group, 2))
#             cannot_link += group
#             group = []
#
#     group = list(itertools.combinations(group, 2))
#     cannot_link += group
#     return cannot_link
#
# def getMergePeakLists(basepath):
#     pL = []
#     for entry in os.listdir(basepath):
#         if os.path.isfile(os.path.join(basepath, entry)):
#             pL1 = pd.read_csv(basepath+entry,low_memory=False, sep="\t", comment='#')
#
#             pL.append(pL1)
#
#
#     return pd.concat(pL).reset_index()
#
# TotPeakList = getMergePeakLists('Data/peax_data/testing_peax/')
# cannot_link = getLinkageConstrain(TotPeakList)
# peaksCoordinates = TotPeakList[['t','r']].to_numpy()
# clusters, centers = cop_kmeans(dataset=peaksCoordinates, k=25, cl=cannot_link)
#
# print(clusters)
#
# print(len(peaksCoordinates))


# fig, ax = plt.subplots()
# ax.scatter(peaksCoordinates.transpose()[0] , peaksCoordinates.transpose()[1])
# for i, txt in enumerate(clusters):
#     ax.annotate(txt,  (peaksCoordinates.transpose()[0][i] , peaksCoordinates.transpose()[1][i]))
#
# plt.show()
