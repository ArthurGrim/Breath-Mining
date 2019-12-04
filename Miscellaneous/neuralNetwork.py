from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def getIntensityMatrixAndLabels(peakListsFolder,peakAlignmentFile,distanceThreshold = 5):
    peakListsFolder = "Data/training_peakIdentification"

    PA = pd.read_csv("HeavyPeakList.txt",low_memory=False, sep="\t", comment='#')
    peakIndices = list(PA.index.values)
    peakCoor = PA[['t','r']].to_numpy().tolist()




    fileNamesList = []
    PLCoordinates = []
    PLCIntensities = []

    for entry in sorted(os.listdir(peakListsFolder)): #for every object in the directory
        if os.path.isfile(os.path.join(peakListsFolder, entry)): #is it is a file
            if "lock" not in entry: #filter out locked files
                PL = pd.read_csv(peakListsFolder+"/"+entry,low_memory=False, sep="\t", comment='#')
                PLCoordinates.append(PL[['t','r']].to_numpy().tolist())
                PLCIntensities.append(PL['signal'].to_numpy().tolist())
                fileNamesList.append(PL.to_numpy()[1][0])

    labels = []
    labelsF = pd.read_csv("Data/labelsTraining.csv",low_memory=False, sep="\t", comment='#')
    for f in fileNamesList:
        r = labelsF.loc[labelsF['file'] == f]
        labels.append(r.to_numpy().tolist()[0][1])

    intensityMatrix = []

    for peakIndex in range(len(peakCoor)): #for each peak in the peak alignement file

        row = []
        for listIndex in range(len(fileNamesList)):

            shortestDist = 10000
            shortestDistIndex = -1
            for p in range(len(PLCoordinates[listIndex])):

                dist = distance.euclidean(peakCoor[peakIndex],[PLCoordinates[listIndex][p][0]*100,PLCoordinates[listIndex][p][1]])

                if (dist < distanceThreshold and shortestDist > dist):
                    shortestDist = dist
                    shortestDistIndex = p

            if shortestDist < 10000:
                row.append(PLCIntensities[listIndex][shortestDistIndex])
            else:
                row.append(0)

        intensityMatrix.append(row)

    return np.array(intensityMatrix).transpose().tolist(), labels



intensityMatrix, labels = getIntensityMatrixAndLabels("Data/training_peakIdentification","HeavyPeakList.txt",5)

scaler = StandardScaler()
scaler.fit(intensityMatrix)
intensityMatrix = scaler.transform(intensityMatrix)


sns.heatmap(intensityMatrix, yticklabels= labels)
plt.show()




clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
scores = cross_val_score(clf, intensityMatrix, labels, cv=5)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))


clfFinal = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
clfFinal.fit(intensityMatrix, labels)
