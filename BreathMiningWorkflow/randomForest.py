import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from scipy.spatial import distance
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import sys

basePath = "Data/"
outPath = "out/"
peaxDataPath = basePath + "training_data_peakLists/" #candy_peax_processed/candy_peax
classLabelsPath = basePath + "labels_training_data.csv"
testingPeaxPath = basePath + "testing_peakIdentification/"
testingPeakAlignmentsPath = basePath + "testing_data_peakLists/"
peakAlignmentsPath = "peakAlignment.csv"


def getIndicatorMatrixAndLabels(peakListsFolder,peakAlignmentFile, labelsFile = classLabelsPath, distanceThreshold = 5):
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
    
    IM = np.array(IM)
    
    return IM.transpose(), labels
   

def getAllCsvAsDataFrame(path, sep):
    dfs = []
    for entry in os.listdir(path):
        filePath = os.path.join(path, entry)
        if os.path.isfile(filePath) and filePath.endswith(".csv"):
            data = pd.read_csv(path + entry, sep=sep, low_memory=False, comment='#')
            dfs.append(data)
    return pd.concat(dfs, sort=False)


def showMoreImportantFeaturesTree(clf, X_train, y_train):
     # extract 5 more important features
    sfm = SelectFromModel(clf, threshold=-np.inf, max_features=5)
    sfm.fit(X_train, y_train)
    
    feature_labels = np.array([])
    for i in range(X_train.shape[1]):
        feature_labels = np.append(feature_labels, "peak " + str(i))
    
    print("most important features:")
    for feature_list_index in sfm.get_support(indices=True):
        print(feature_labels[feature_list_index])
    
    important_indexes = np.array((sfm.get_support(indices=True)))
    feature_important_labels = feature_labels[important_indexes]
    X_important_train = sfm.transform(X_train)
    
    dot_data = StringIO()
    
    important_features_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    important_features_clf.fit(X_important_train, y_train)
    
    estimator = important_features_clf.estimators_[15]
    export_graphviz(estimator, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, 
                feature_names = feature_important_labels,
                class_names=np.unique(y_train))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('decisionTreeImportantFeatures.png')
    Image(graph.create_png())


def main(peakListsFolder = peaxDataPath, peakAlignmentFile = peakAlignmentsPath, classLabels = classLabelsPath, testPeaksList = testingPeakAlignmentsPath, distanceThreshold=5):
    
    X_train, y_train = getIndicatorMatrixAndLabels(peakListsFolder,peakAlignmentFile,classLabels)
    sns.heatmap(X_train, yticklabels=y_train)
    plt.show()
    
    if (input("Perform 5-fold cross-validation (Y/N)").upper() == "Y"):

        print(" Performing 5-fold cross-validation with training data \n" )
    
        clf = RandomForestClassifier(n_estimators=1000, random_state=0)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    
    if (input("Generate confussion matrix (Y/N)").upper() == "Y"):
        ytestTotal = np.array([])
        ypredTotal = np.array([])
        for i in range(50):
            X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=1000, random_state=0)
            clf.fit(X_train2, y_train2)
            y_pred = clf.predict(X_test2)
            
            ytestTotal = np.append(y_test2, ytestTotal)
            ypredTotal = np.append(y_pred, ypredTotal)
        
        # show confusion matrix
        print("Confusion matrix with test size = 0.3: \n")
        print(confusion_matrix(ytestTotal, ypredTotal))
    
    # Create Mtest matrix
    testPeakData = getAllCsvAsDataFrame(testPeaksList, '\t').to_numpy()
    X_test, _ = getIndicatorMatrixAndLabels(testPeaksList,peakAlignmentFile)
    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    
    # apply the random forest classifier
    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)   
    fileNames = np.unique(testPeakData[:,0])
    
    
    showMoreImportantFeaturesTree(clf, X_train, y_train)
    
    outputFile = input("Save prediction as:  ")
    f = open(outputFile, "w")
    for i in range(len(fileNames)):
        f.write("{}\t{}\n".format(fileNames[i],y_pred[i]))
    f.close()
    
    print("Prediction has been saved as: " + outputFile)

# uncomment to use default values for directories
#main()
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
