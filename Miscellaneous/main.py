import os
import tkinter.filedialog as fd

from tkinter import Tk
Tk().withdraw()

def oneClickClassifier():
    print("not ready")

def launchPeakDetection():
    print("not ready")

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
    print("0 = Exit\n1 = none\n2 = Perform Peak Alignement")
    choice = int(input("select option: "))

    if(choice==0):
        run = False
        print("Exiting program")
    elif(choice==1):
        print("1")
    elif(choice==2):
        launchPeakAlignment()
    else:
        print("invalid input")
