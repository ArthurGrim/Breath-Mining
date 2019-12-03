import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

basePath = "Data/"
peaxDataDirectory = basePath + "peax_data/candy_peax/"
peakAlignmentsPath = basePath + "peaksList.txt"
fileCategoriesPath = basePath + "class_labels.txt"


def getPeaxDataFrame():
    dfs = []
    for entry in os.listdir(peaxDataDirectory):
        filePath = os.path.join(peaxDataDirectory, entry)
        if os.path.isfile(filePath) and filePath.endswith(".csv"):
            data = pd.read_csv(peaxDataDirectory + entry, sep="\t", low_memory=False, comment='#')
            dfs.append(data)
    return pd.concat(dfs)


def comparePeeks(t1, t2, r1, r2):
    try:
        dift = t1 - t2
        difr = r1 - r2
    except:
        return False
    return (abs(dift) < 0.03) & (abs(difr) < (3.0 + r1 * 0.1))


peakData = getPeaxDataFrame().to_numpy()
peakAlignments = pd.read_csv(peakAlignmentsPath, sep='\t').to_numpy()

fileNames, indices = np.unique(peakData[:,0], return_inverse=True)
Mtrain = pd.DataFrame(np.zeros((peakAlignments.shape[0], 6)), columns=fileNames)
i = 0
for peak in peakAlignments:
    trainIndices = []
    for peak2 in peakData:
        if comparePeeks(peak[0], peak2[2]*100, peak[1], peak2[3]):
            trainIndices.append(peak2[0])
    for index in trainIndices:
        Mtrain[index][i] = 1
    i+=1

# Test indicator matrix
ax = sns.heatmap(Mtrain)
ax
transpose = Mtrain.transpose()
# add row with category names
fileCategories = pd.read_csv(fileCategoriesPath, sep='\t')
numRows = transpose.shape[0]
numColumns = transpose.shape[1]

transpose.insert(numColumns, numColumns, fileCategories['candy'].to_numpy())

transpose.to_csv(path_or_buf = 'indicatorMatrix.csv')

X_train = transpose.iloc[:, 0:numColumns-1].values
y_train = transpose.iloc[:, -1].values


a, X_test, b, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test = labelencoder.fit_transform(y_test)
print(y_train)

regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))