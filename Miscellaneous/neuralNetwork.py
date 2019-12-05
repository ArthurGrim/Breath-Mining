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
import sys
import tkinter.filedialog as fd


def getIntensityMatrixAndLabels(peakListsFolder,peakAlignmentFile,distanceThreshold = 5):

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
        if(r.empty == False):
            labels.append(r.to_numpy().tolist()[0][1])
        else:
            labels.append(f)

    IM = []

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

        IM.append(row)

    return np.array(IM).transpose().tolist(), labels


def main(peakListsFolder,peakAlignmentFile,distanceThreshold=5):

    intensityMatrix, labels = getIntensityMatrixAndLabels(peakListsFolder,peakAlignmentFile,distanceThreshold)

    scaler = StandardScaler()
    scaler.fit(intensityMatrix)
    intensityMatrix = scaler.transform(intensityMatrix)


    sns.heatmap(intensityMatrix, yticklabels= labels)
    plt.show()




    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
    scores = cross_val_score(clf, intensityMatrix, labels, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))


    clfFinal = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
    clfFinal.fit(intensityMatrix, labels)

    print("Select directory of peaks lists for prediction")
    predictPeakLists = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of peaks lists for prediction")
    predictPeakLists = predictPeakLists.replace(os.getcwd()+"/","")+"/"
    print("Selected Dir: "+predictPeakLists)

    testingX, testingY  = getIntensityMatrixAndLabels(predictPeakLists,peakAlignmentFile,distanceThreshold)
    print(len(testingX))
    print(testingY)
    scaler.fit(testingX)
    testingX = scaler.transform(testingX)


    prediction = clfFinal.predict(testingX)
    print(prediction)


    outputFile = input("Save prediction as:  ")
    f = open(outputFile, "w")

    f.write("\"t\"\t\"r\"\n")
    for i in range(len(prediction)):
        f.write("{}\t{}\n".format(testingY[i],prediction[i]))
    f.close()






if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
