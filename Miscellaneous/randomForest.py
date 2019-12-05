import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

basePath = "Data/"
outPath = "out/"
peaxDataDirectory = basePath + "training_peakIdentification/" #candy_peax_processed/candy_peax
fileCategoriesPath = basePath + "labels_training.csv"
testingPeaxPath = basePath + "testing_peakIdentification/"
testingPeakAlignmentsPath = basePath + "testing_peakList.csv"
peakAlignmentsPath = basePath + "peaksList.csv"
indicatorMatrixPath = outPath + "indicatorMatrix.csv"
predictedLabelsPath = outPath + "predictedLabels.csv"

def getIntensityMatrixAndLabels(peakListsFolder, peakAlignmentFile, isTraining = 1,distanceThreshold = 5):

    PA = pd.read_csv(peakAlignmentFile,low_memory=False, sep="\t", comment='#')
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
    labelsF = pd.read_csv(fileCategoriesPath,low_memory=False, sep="\t", comment='#')
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

def getAllCsvAsDataFrame(path, sep):
    dfs = []
    for entry in os.listdir(path):
        filePath = os.path.join(path, entry)
        if os.path.isfile(filePath) and filePath.endswith(".csv"):
            data = pd.read_csv(path + entry, sep=sep, low_memory=False, comment='#')
            dfs.append(data)
    return pd.concat(dfs, sort=False)

Mtrain, labels = getIntensityMatrixAndLabels(peaxDataDirectory,peakAlignmentsPath)

scaler = StandardScaler()
scaler.fit(Mtrain)
Mtrain = scaler.transform(Mtrain)

sns.heatmap(Mtrain, yticklabels= labels)
plt.show()

# transpose the matrix to obtain the X matrix
transpose = Mtrain.transpose()

# separate x and y
X_train = Mtrain
y_train = labels

# encode y value to decimal values
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)

# Create Mtest matrix
testPeakAlignments = pd.read_csv(testingPeakAlignmentsPath, sep='\t').to_numpy()
testPeakData = getAllCsvAsDataFrame(testingPeaxPath, '\t').to_numpy()
Mtest, labels = getIntensityMatrixAndLabels(testingPeaxPath,peakAlignmentsPath)

scaler = StandardScaler()
scaler.fit(Mtest)
Mtest = scaler.transform(Mtest)

ax = sns.heatmap(Mtest)
ax

X_test = Mtest

# apply the random forest classifier
regressor = RandomForestClassifier(n_estimators=1000, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred = list(labelencoder.inverse_transform(y_pred))
fileNames = np.unique(testPeakData[:,0])

f = open(predictedLabelsPath, "w")
for i in range(len(fileNames)):
    f.write("{}\t{}\n".format(fileNames[i],y_pred[i]))
f.close()
