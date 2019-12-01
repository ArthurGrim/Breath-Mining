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
    os.system("python3 peakAlignments.py " + rawDir + " " + peaksDir )

def launchIndicatorMatrix():
    print("not ready")

run = True
while(run == True):
    print("### MAIN MENU ###")
    print("0 = Exit\n1 = Perform Peak Identification (via PEAX)\n2 = Perform Peak Alignement")
    choice = int(input("select option: "))

    if(choice==0):
        run = False
        print("Exiting program")
    elif(choice==1):
        launchPeakDetection()
    elif(choice==2):
        launchPeakAlignment()
    else:
        print("invalid input")
