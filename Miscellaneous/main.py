import os
import tkinter.filedialog as fd

from tkinter import Tk
Tk().withdraw()

def oneClickClassifier():
    print("not ready")

def launchPeakDetection():
    # print("Select directory of raw IMS/MMC file")
    # rawDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of raw IMS/MMC file")
    # print("Selected Dir: "+rawDir)
    #
    # print("Select directory an output dire1ctory to store peak lists")
    # peaksDir = fd.askdirectory(initialdir = os.getcwd(),title = "Select directory of peakLists")
    # print("Selected Dir: "+peaksDir)
    #
    # print("Select PEAX parameter file (.cfg)")
    # paramFile = fd.askopenfilename(initialdir = os.getcwd(),title = "Select PEAX parameter file (.cfg)")
    # print("Selected Dir: "+paramFile)

    ipD = fd.askdirectory()
    opD = fd.askdirectory()
    paF = fd.askopenfilename()

    os.system("python3 peakIdentification.py " + ipD.replace(os.getcwd()+"/","")+"/" + " " + opD.replace(os.getcwd()+"/","")+"/" + " " + paF.replace(os.getcwd()+"/","")+"/" )

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

    print("Select peak alignment file")
    peakAlignmentFile = fd.askopenfilename(initialdir = os.getcwd(),title = "Select peak alignment file")
    peakAlignmentFile = peakAlignmentFile.replace(os.getcwd()+"/","")
    print("Selected Dir: "+peakAlignmentFile)

    thresh = input("Maximal peak detection distance (default = 5) : ")
    os.system("python3 neuralNetwork.py " + peaksDir + " " + peakAlignmentFile + " " + thresh )

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

run = True
while(run == True):
    print("### MAIN MENU ###")
    print("0 = Exit\n1 = Perform Peak Identification (via PEAX)\n2 = Perform Peak Alignement\n3 = Train an MLP classifier and predict\n4 = Train a Random Forest and predict")
    choice = int(input("select option: "))

    if(choice==0):
        run = False
        print("Exiting program")
    elif(choice==1):
        launchPeakDetection()
    elif(choice==2):
        launchPeakAlignment()
    elif(choice==3):
        launchNeuralNetwork()
    elif(choice==4):
        launchRandomForest()
    else:
        print("invalid input")
