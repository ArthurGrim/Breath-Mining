import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import sys

basePath = "Data/"
outPath = "out/"
peaxDataPath = basePath + "training_data_peakLists/" #candy_peax_processed/candy_peax
fileCategoriesPath = basePath + "labels_training_data.csv"
testingPeaxPath = basePath + "testing_peakIdentification/"
testingPeakAlignmentsPath = basePath + "testing_data_peakLists/"
peakAlignmentsPath = "peakAlignment.csv"


def getIndicatorMatrixAndLabels(peakListsFolder,peakAlignmentFile,distanceThreshold = 5,isTest = False, labelsFile= "Data/labels_training_data.csv"):
    # Return an matrix wich rows corresponds to one peak list to peak list folder and rows to to the signal of a peak for each peak coordinates in the peak alignement file
    # and the class label of each peak list in the same order as they appear in the matrix
    print("\n\n -=Generating Density Matrix=- \n\n" )

    PA = pd.read_csv(peakAlignmentFile,low_memory=False, sep="\t", comment='#')
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
    labelsF = pd.read_csv(labelsFile,low_memory=False, sep="\t", comment='#')
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
            for p in range(len(PLCoordinates[listIndex])):

                dist = distance.euclidean(peakCoor[peakIndex],[PLCoordinates[listIndex][p][0]*100,PLCoordinates[listIndex][p][1]])

                if (dist < distanceThreshold and shortestDist > dist):
                    shortestDist = dist

            if shortestDist < 10000:
                row.append(1)
            else:
                row.append(0)

        IM.append(row)
    
    giniIndexes = np.array([])
    IM = np.array(IM)
    if isTest == False:  
        giniIndexes = getGiniTopDiscriminatingIndexes(IM,labels)
    
    return IM, labels, giniIndexes

def getGiniTopDiscriminatingIndexes(IM,labels):
    peakFHalls = np.array([np.zeros(IM.shape[0])])
    peakFCitrus = np.array([np.zeros(IM.shape[0])])
    
    for i in range(len(labels)):
        if labels[i] == 'halls':
            peakFHalls = np.sum([peakFHalls, IM[:,i]], axis=0)
        elif labels[i] == 'citrus':
            peakFCitrus = np.sum([peakFCitrus, IM[:,i]], axis=0)
    
    frequencies = np.sum([peakFHalls,peakFCitrus], axis=0)
    peakFHalls = np.divide(peakFHalls,frequencies)
    peakFCitrus = np.divide(peakFCitrus,frequencies)
    giniIndexes = np.argsort(np.multiply(peakFHalls,peakFCitrus))[0,-5:]
    
    return giniIndexes
   

def getAllCsvAsDataFrame(path, sep):
    dfs = []
    for entry in os.listdir(path):
        filePath = os.path.join(path, entry)
        if os.path.isfile(filePath) and filePath.endswith(".csv"):
            data = pd.read_csv(path + entry, sep=sep, low_memory=False, comment='#')
            dfs.append(data)
    return pd.concat(dfs, sort=False)

def main(peakListsFolder = peaxDataPath, peakAlignmentFile = peakAlignmentsPath, testPeaksList = testingPeakAlignmentsPath, distanceThreshold=5):
    Mtrain, labels, giniIndexes = getIndicatorMatrixAndLabels(peakListsFolder,peakAlignmentFile)
    Mgini = Mtrain[giniIndexes].transpose().tolist()
    Mtrain = Mtrain.transpose().tolist()
    sns.heatmap(Mtrain, yticklabels=labels)
    plt.show()
    
    if (input("Perform 5-fold cross-validation (Y/N)").upper() == "Y"):

        print(" Performing 5-fold cross-validation with training data \n" )
    
        clf = RandomForestClassifier(n_estimators=1000, random_state=0)
        scores = cross_val_score(clf, Mtrain, labels, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    
    if (input("Generate confussion matrix (Y/N)").upper() == "Y"):
        ytestTotal = np.array([])
        ypredTotal = np.array([])
        for i in range(50):
            X_train, X_test, y_train, y_test = train_test_split(Mtrain, labels, test_size=0.8, random_state=42)
            clf = RandomForestClassifier(n_estimators=1000, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            ytestTotal = np.append(y_test, ytestTotal)
            ypredTotal = np.append(y_pred, ypredTotal)
        
        # show confusion matrix
        print("Confusion matrix with test size = 0.3: \n")
        print(confusion_matrix(ytestTotal, ypredTotal))
    
    # encode y value to decimal values
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels)
    
    # Create Mtest matrix
    testPeakData = getAllCsvAsDataFrame(testPeaksList, '\t').to_numpy()
    X_test, _, _ = getIndicatorMatrixAndLabels(testPeaksList,peakAlignmentFile,distanceThreshold,True)
    X_test_gini = X_test[giniIndexes].transpose().tolist()
    X_test = X_test.transpose().tolist()
    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    
    # apply the random forest classifier
    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    clf.fit(Mtrain, labels)
    y_pred = clf.predict(X_test)
    y_pred = list(labelencoder.inverse_transform(y_pred))
    
    fileNames = np.unique(testPeakData[:,0])
    
    # plot gini decision tree
    dtreeClf = tree.DecisionTreeClassifier()
    tree.plot_tree(dtreeClf.fit(Mgini,labels)) 
    
    outputFile = input("Save prediction as:  ")
    f = open(outputFile, "w")
    for i in range(len(fileNames)):
        f.write("{}\t{}\n".format(fileNames[i],y_pred[i]))
    f.close()
    
    print("Prediction has been saved as: " + outputFile)

# uncomment to use default values for directories
# main()
if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
