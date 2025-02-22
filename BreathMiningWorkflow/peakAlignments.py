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

palette = cycle(sns.color_palette().as_hex())



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



def main(rawDir, peaksDir, distanceThreshold):
    ###### Perform Hierachical clustering on raw data ##########
    distanceThreshold = int(distanceThreshold)#minimal distance for pairing 2 points
    print("---==Perfoming hierchical clustering==--- \n")
    M,fileNames = getRawDataMatrix(rawDir)
    linked = linkage(M, 'single')
    alignementOrder = np.delete(np.array(linked),np.s_[2,3],1).astype(int) #Get the pair of peak lists to align and in which order to align

    #
    # dendrogram(linked)
    # plt.show()

    print("\n--==Performing pairwise peak alignment==--\n")

    nodesAlignements = {} #dictionary containing the alignement for each leaf and node of the tree obtained via Hierachical clustering

    alignementIndex = len(fileNames)-1 #dictionnary key of the new peak lists resulting of the alignement of two peak list

    for p in range(len(fileNames)):
        pl = pd.read_csv(peaksDir+fileNames[p],low_memory=False, sep="\t", comment='#')[['t','r']]
        pl["t"] *= 100 #To normalize the "r" and "t" axis
        pl = pl.to_numpy().tolist()
        nodesAlignements[p] = pl


    m = Munkres() # create a Munkres object

    for pairs in alignementOrder: #for each node in the tree from bottom to top

        unusedPeaks = list(range(len(nodesAlignements[pairs[1]])))
        usedPeaks = []
        if len(nodesAlignements[pairs[0]]) > len(nodesAlignements[pairs[1]]): # The matrix has to has more row than collumns
            pairs = [pairs[1],pairs[0]]

        alignementIndex+=1
        nodesAlignements[alignementIndex] = []
        distMat = distance_matrix(nodesAlignements[pairs[0]], nodesAlignements[pairs[1]])


        indexes = m.compute(distMat)
        nPaired = 0
        for row, column in indexes:

            dist = distance.euclidean(nodesAlignements[pairs[0]][row],nodesAlignements[pairs[1]][column])
            usedPeaks.append(column)
            if dist<distanceThreshold:
                nPaired += 1
                midpointCoordinates = midpoint(nodesAlignements[pairs[0]][row],nodesAlignements[pairs[1]][column])
                nodesAlignements[alignementIndex].append(midpointCoordinates)
            else:
                nodesAlignements[alignementIndex].append(nodesAlignements[pairs[0]][row])
                nodesAlignements[alignementIndex].append(nodesAlignements[pairs[1]][column])


        for i in unusedPeaks[:]:
            if i in usedPeaks:
                unusedPeaks.remove(i)

        for j in unusedPeaks: #Adding the peaks that haven't been paired previously
            nodesAlignements[alignementIndex].append(nodesAlignements[pairs[1]][j])

        print("\nPeaklist {} (lenght = {}) and {} (lenght = {}) have been aligned: \nPeaks paired: {}\nTotal peaks in aligned peaklist: {}\n ".format(pairs[0],len(nodesAlignements[pairs[0]]),pairs[1],len(nodesAlignements[pairs[1]]),nPaired,len(nodesAlignements[alignementIndex])))
        plt.scatter(np.array(nodesAlignements[pairs[0]]).transpose()[0], np.array(nodesAlignements[pairs[0]]).transpose()[1], c=next(palette),marker=".")
        plt.scatter(np.array(nodesAlignements[pairs[1]]).transpose()[0], np.array(nodesAlignements[pairs[1]]).transpose()[1], c=next(palette),marker=".")



    finalAlignement = max(nodesAlignements, key=int) # key of the final alignement of all the peaklists

    if (input("Print plot ? (Y/N)") == "Y"):
        plt.scatter(np.array(nodesAlignements[finalAlignement]).transpose()[0], np.array(nodesAlignements[finalAlignement]).transpose()[1], c="black",marker = "x" )
        plt.show()


    outputFile = input("Save final aligned peaks list as:  ")
    f = open(outputFile, "w")

    f.write("\"t\"\t\"r\"\n")
    for i in range(len(nodesAlignements[finalAlignement])):
        f.write("\"{}\"\t{}\t{}\n".format(i,round(nodesAlignements[finalAlignement][i][0],3), round(nodesAlignements[finalAlignement][i][1],3)))
    f.close()




if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2], sys.argv[3])
