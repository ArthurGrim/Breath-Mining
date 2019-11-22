import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile





def peaxCommandMaker(inputFile, outputFile, paramFile):
    inputFile = inputFile.split("/")[-1]
    paramFile = paramFile.split("/")[-1]
    os.popen("chmod 775 " + paramFile) #allow file to be read/open
    return "./peax " + inputFile + " " + outputFile +" ./-p " + paramFile + "/"




Tk().withdraw()
ipF = askopenfilename()
opF = "outputPeaks.txt"
paF = askopenfilename()



cmd = peaxCommandMaker(ipF, opF, paF)
print(cmd)


f = os.popen(cmd)
now = f.read()

print ("Results from peak Identification:", now)
