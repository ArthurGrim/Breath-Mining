import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

basePath = "Data/"
outPath = "out/"
peaxDataDirectory = basePath + "training_peakIdentification/" #candy_peax_processed/candy_peax
fileCategoriesPath = basePath + "labels_training.csv"
testingPeaxPath = basePath + "testing_peakIdentification/"
testingPeakAlignmentsPath = basePath + "testing_peakList.csv"
peakAlignmentsPath = basePath + "peaksList.csv"
indicatorMatrixPath = outPath + "indicatorMatrix.csv"
predictedLabelsPath = outPath + "predictedLabels.csv"


def getAllCsvAsDataFrame(path, sep):
    dfs = []
    for entry in os.listdir(path):
        filePath = os.path.join(path, entry)
        if os.path.isfile(filePath) and filePath.endswith(".csv"):
            data = pd.read_csv(path + entry, sep=sep, low_memory=False, comment='#')
            dfs.append(data)
    return pd.concat(dfs, sort=False)


def comparePeeks(t1, t2, r1, r2):
    try:
        dift = t1 - t2
        difr = r1 - r2
    except:
        return False
    return (abs(dift) < 0.03) & (abs(difr) < (3.0 + r1 * 0.1))

peakDataFrame = getAllCsvAsDataFrame(peaxDataDirectory, '\t')
peakAlignmentsDataFrame = pd.read_csv(peakAlignmentsPath, sep='\t')
peakData = peakDataFrame.to_numpy()
peakAlignments = peakAlignmentsDataFrame.to_numpy()


def createIndicatorMatrix(alignments, data):
    fileNames, indices = np.unique(data[:,0], return_inverse=True)
    indicatorMatrix = pd.DataFrame(np.zeros((peakAlignments.shape[0], len(fileNames))), columns=fileNames)
    i = 0
    for peak in alignments:
        indices = []
        for peak2 in data:
            if comparePeeks(peak[0], peak2[2]*100, peak[1], peak2[3]):
                indices.append(peak2[0])
        for index in indices:
            indicatorMatrix[index][i] = 1
        i+=1
        
    return indicatorMatrix
     

Mtrain = createIndicatorMatrix(peakAlignments, peakData)
fileCategoriesNp = pd.read_csv(fileCategoriesPath, sep='\t').to_numpy()


citrusMatrix = pd.DataFrame()
originalMatrix = pd.DataFrame()

classes, indexes = np.unique(fileCategoriesNp[:,1], return_inverse=True)

for c in classes:
    print(c)
    indexes = np.where(fileCategoriesNp[:,1] == c)
    for j in indexes:
        fileName = fileCategoriesNp[j,0]
        print(fileName)
        if c == 'citrus':
            citrusMatrix = citrusMatrix.append(Mtrain[fileName])
        elif c == 'halls':
            originalMatrix = originalMatrix.append(Mtrain[fileName])
    
    
ax = sns.heatmap(citrusMatrix)
ax.set_title('citrus heat map')
ax

bx = sns.heatmap(originalMatrix)
bx.set_title('halls heat map')
bx

# transpose the matrix to obtain the X matrix
transpose = Mtrain.transpose()

# add row with category names (y)
numRows = transpose.shape[0]
numColumns = transpose.shape[1]
transpose.insert(numColumns, numColumns, fileCategories['candy'].to_numpy())
transpose.to_csv(path_or_buf = indicatorMatrixPath)

# separate x and y
X_train = transpose.iloc[:, 0:numColumns-1].values
y_train = transpose.iloc[:, -1].values

# encode y value to decimal values
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)

# Create Mtest matrix
testPeakAlignments = pd.read_csv(testingPeakAlignmentsPath, sep='\t').to_numpy()
testPeakData = getAllCsvAsDataFrame(testingPeaxPath, '\t').to_numpy()
Mtest = createIndicatorMatrix(testPeakAlignments, testPeakData)

ax = sns.heatmap(Mtest)
ax

X_test = Mtest.transpose()
X_test = X_test.iloc[:, :-1]

# apply the random forest classifier
regressor = RandomForestClassifier(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred = list(labelencoder.inverse_transform(y_pred))
predDict = {"raw_file_name": list(X_test.index.values), "class_label": y_pred}
predDf = pd.DataFrame.from_dict(predDict)
predDf.to_csv(path_or_buf = predictedLabelsPath, sep='\t', index=False)
