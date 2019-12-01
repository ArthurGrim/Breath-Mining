import numpy as np
import sklearn
import math
import itertools
import pandas as pd
import os
import sys
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from munkres import Munkres
from scipy.spatial import distance
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

from itertools import cycle
cycol = cycle('bgrcmk')



def getRawDataAslist(filename):
#Input: IMC/MMS raw file (has to be a .csv file with "," as separator)
#Read the matrix contain in the file and concatenate rows in a single list
#Output: List
    print("Reading: " + filename )
    rawMat = pd.read_csv(filename,low_memory=False, sep=",", comment='#')
    rawMat = rawMat.drop(rawMat.index[[0]])
    rawMat = rawMat.drop(rawMat.columns[[0,1,2]], axis=1)
    rawMat = rawMat.to_numpy()
    rawList = rawMat.flatten()
    return rawList.tolist()


def getRawDataMatrix(basepath):
#Input: The path to a folder containing IMC/MMS raw file
#Use "getRawDataAslist" to generate a matrix with the datalist from one file as a row and one row for each files
#Output: list of lists, and name of the files
    mat = []
    fileNames = []
    for entry in sorted(os.listdir(basepath)): #for every object in the directory
        if os.path.isfile(os.path.join(basepath, entry)): #is it is a file
            if "lock" not in entry: #filter out locked files
                mat.append(getRawDataAslist(basepath+entry))
                fileNames.append(entry)
    return mat,fileNames


def midpoint(p1, p2):
#Input: two set of coordinates in a bi-dimentional space (can be list or tuples)
#Return: coordinates of the midpoint
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]



def main(rawDir, peaksDir):
    ###### Perform Hierachical clustering on raw data ##########

    print("---==Perfoming hierchical clustering==--- \n")
    M,fileNames = getRawDataMatrix(rawDir)
    linked = linkage(M, 'single')
    alignementOrder = np.delete(np.array(linked),np.s_[2,3],1).astype(int) #Get the pair of peak lists to align and in which order to align




    # plt.figure(figsize=(10, 7))
    # dendrogram(linked, labels=range(0,6))
    # plt.show()

    print("\n--==Performing pairwise peak alignment==--\n")

    nodesAlignements = {} #dictionary containing the alignement for each leaf and node of the tree obtained via Hierachical clustering
    distanceThreshold = 3 #minimal distance for pairing 2 points
    alignementIndex = len(fileNames)-1 #dictionnary key of the new peak lists resulting of the alignement of two peak list

    for p in range(len(fileNames)):
        pl = pd.read_csv(peaksDir+fileNames[p],low_memory=False, sep="\t", comment='#')[['t','r']]
        pl["t"] *= 100 #To normalize the "r" and "t" axis
        pl = pl.to_numpy().tolist()
        nodesAlignements[p] = pl


    m = Munkres() # create a Munkres object

    for pairs in alignementOrder: #for each node in the tree from bottom to top


        if len(nodesAlignements[pairs[0]]) > len(nodesAlignements[pairs[1]]): # The matrix has to has more row than collumns
            pairs = [pairs[1],pairs[0]]

        alignementIndex+=1
        nodesAlignements[alignementIndex] = []
        distMat = distance_matrix(nodesAlignements[pairs[0]], nodesAlignements[pairs[1]])


        indexes = m.compute(distMat)
        nPaired = 0
        for row, column in indexes:
            dist = distance.euclidean(nodesAlignements[pairs[0]][row],nodesAlignements[pairs[1]][column])

            if dist<distanceThreshold:
                nPaired += 1
                midpointCoordinates = midpoint(nodesAlignements[pairs[0]][row],nodesAlignements[pairs[1]][column])
                nodesAlignements[alignementIndex].append(midpointCoordinates)
            else:
                nodesAlignements[alignementIndex].append(nodesAlignements[pairs[0]][row])
                nodesAlignements[alignementIndex].append(nodesAlignements[pairs[1]][column])

        print("\nPeaklist {} and {} have been aligned: \nPeaks paired: {} Total peaks in aligned peaklist: {}\n ".format(pairs[0],pairs[1],nPaired,len(nodesAlignements[alignementIndex])))
        plt.scatter(np.array(nodesAlignements[pairs[0]]).transpose()[0], np.array(nodesAlignements[pairs[0]]).transpose()[1], c=next(cycol),marker=".")
        plt.scatter(np.array(nodesAlignements[pairs[1]]).transpose()[0], np.array(nodesAlignements[pairs[1]]).transpose()[1], c=next(cycol),marker=".")



    if (input("Print plot ? (Y/N)") == "Y"):
        plt.scatter(np.array(nodesAlignements[10]).transpose()[0], np.array(nodesAlignements[10]).transpose()[1], c="red",marker = "X" )
        plt.show()

    # sns.heatmap(distMat)
    # plt.show()


    outputFile = input("Save final aligned peaks list as:  ")
    f = open(outputFile, "w")

    finalAlignement = max(nodesAlignements, key=nodesAlignements.get) # key of the final alignement of all the peaklists

    f.write("\"t\" \"r\"\n")
    for i in range(len(nodesAlignements[finalAlignement])):
        f.write("\"{}\"\t{} {}\n".format(i,round(nodesAlignements[finalAlignement][i][0],3), round(nodesAlignements[finalAlignement][i][1],3)))
    f.close()




if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
