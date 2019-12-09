import os
import tkinter.filedialog as fd

from tkinter import Tk
Tk().withdraw()

def oneClickClassifier():
    print("not ready")

def launchDensityPlot():

    print("Select raw IMS file:")
    paF = fd.askopenfilename(initialdir = os.getcwd(),title = "Select raw IMS file")

    os.system("python3 densityPlot.py " + paF.replace(os.getcwd()+"/","") )

def launchPeakDetection():

    print("Select directory of raw IMS/MMC file")
    ipD = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of raw IMS/MMC file")
    print("Select directory an output dire1ctory to store peak lists")
    opD = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory an output dire1ctory to store peak lists")
    print("Select PEAX parameter file (.cfg)")
    paF = fd.askopenfilename(initialdir = os.getcwd(),title = "Select PEAX parameter file (.cfg)")

    os.system("python3 peakIdentification.py " + ipD.replace(os.getcwd()+"/","")+"/" + " " + opD.replace(os.getcwd()+"/","")+"/" + " " + paF.replace(os.getcwd()+"/","")+"/" )

def launchRandomForest():
    print("Select directory of training peak lists")
    peaksDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of training peak lists")
    peaksDir = peaksDir.replace(os.getcwd()+"/","")+"/"
    print("Selected Dir: "+peaksDir)

    print("Select peak alignment file")
    peakAlignmentFile = fd.askopenfilename(initialdir = os.getcwd(),title = "Select peak alignment file")
    peakAlignmentFile = peakAlignmentFile.replace(os.getcwd()+"/","")
    print("Selected Dir: "+peakAlignmentFile)

    print("Select directory of testing peak lists")
    testPeaksDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of testing peak lists")
    testPeaksDir = testPeaksDir.replace(os.getcwd()+"/","")+"/"
    print("Selected Dir: "+testPeaksDir)

    thresh = input("Maximal peak detection distance (default = 5) : ")
    os.system("python3 randomForest.py " + peaksDir + " " + peakAlignmentFile + " " + testPeaksDir + " " + thresh )




def launchPeakAlignment():
    print("Select directory of raw IMS/MMC file")
    rawDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of raw IMS/MMC file")
    rawDir = rawDir.replace(os.getcwd()+"/","")+"/"
    print("Selected Dir: "+rawDir)

    print("Select directory of peak lists")
    peaksDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of peakLists")
    peaksDir = peaksDir.replace(os.getcwd()+"/","")+"/"
    print("Selected Dir: "+peaksDir)

    thresh = input("Minimal distance to merge two peaks (default = 5) : ")
    os.system("python3 peakAlignments.py " + rawDir + " " + peaksDir + " " + thresh )



def launchIndicatorMatrix():
    print("not ready")



def launchNeuralNetwork():
    print("Select directory of peak lists")
    peaksDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of peak lists")
    peaksDir = peaksDir.replace(os.getcwd()+"/","")+"/"
    print("Selected Dir: "+peaksDir)

    print("Select class label file")
    labFi = fd.askopenfilename(initialdir = os.getcwd(),title = "Select class label file")
    labFi = labFi.replace(os.getcwd()+"/","")
    print("Selected Dir: "+labFi)

    print("Select peak alignement file")
    peakAlignmentFile = fd.askopenfilename(initialdir = os.getcwd(),title = "Select peak alignement file")
    peakAlignmentFile = peakAlignmentFile.replace(os.getcwd()+"/","")
    print("Selected Dir: "+peakAlignmentFile)

    thresh = input("Maximal peak detection distance (default = 5) : ")
    os.system("python3 neuralNetwork.py " + peaksDir + " " + peakAlignmentFile + " " + thresh + " " + labFi)

run = True
while(run == True):
    print("### MAIN MENU ###")
    print("0 = Exit\n1 = Density plot from raw file\n2 = Perform Peak Identification (via PEAX)\n3 = Perform Peak Alignement\n4 = Random forest\n5 = Train an MLP classifier and predict")

    while True:
        try:
            choice = int(input("select option: "))
            break
        except ValueError:
            print("Input must be an integer")

    if(choice==0):
        run = False
        print("Exiting program")
    elif(choice==41):
        launchDensityPlot()
    elif(choice==1):
        launchPeakDetection()
    elif(choice==2):
        launchPeakAlignment()
    elif(choice==4):
        launchRandomForest()
    elif(choice==5):
        launchNeuralNetwork()


    else:
        print("invalid input")
