import numpy
import sklearn
from copkmeans.cop_kmeans import cop_kmeans
import itertools
import pandas as pd
import os



import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')



basepath = 'Data/peax_data/testing_peax/'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        print(entry)

def getLinkageConstrain(df):

    cannot_link = []
    group = []
    grpName = df["measurement_name"].iloc[0]
    print(grpName)
    for index, row in df.iterrows():
        if grpName == row["measurement_name"]:
            group.append(index)
        else:
            grpName = row["measurement_name"]
            group = list(itertools.combinations(group, 2))
            cannot_link += group
            group = []

    group = list(itertools.combinations(group, 2))
    cannot_link += group
    return cannot_link

def getMergePeakLists(basepath):
    pL = []
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            pL1 = pd.read_csv(basepath+entry,low_memory=False, sep="\t", comment='#')

            pL.append(pL1)


    return pd.concat(pL).reset_index()




TotPeakList = getMergePeakLists('Data/peax_data/testing_peax/')

print(TotPeakList)

cannot_link = getLinkageConstrain(TotPeakList)

print(cannot_link)

peaksCoordinates = TotPeakList[['t','r']].to_numpy()



clusters, centers = cop_kmeans(dataset=peaksCoordinates, k=25, cl=cannot_link)

print(clusters)

print(len(peaksCoordinates))


fig, ax = plt.subplots()
ax.scatter(peaksCoordinates.transpose()[0] , peaksCoordinates.transpose()[1])
for i, txt in enumerate(clusters):
    ax.annotate(txt,  (peaksCoordinates.transpose()[0][i] , peaksCoordinates.transpose()[1][i]))

plt.show()
