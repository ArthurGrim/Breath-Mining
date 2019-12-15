Breath Mining Workflow
User Guide


Setup

The workflow requires Python (version 3.6.8 or later versions) to be installed on your computer.
In order to run the peak detection function via PEAX, the program needs to be run on Linux. 

Several python packages are required, they can be installed automatically by launching the setup.py script via the following command line:

Python3 setup.py

The installation of packages requires the python module pip3 to be installed on your machine, if it is not the case it can be installed via the following command line (Linux only):

sudo apt install python3-pip

Launching the workflow

To launch the workflow run the main.py script with Python 3. This can be done using the following command in the directory were the script is located. 

Python3 main.py

Main menu

Main menu allows selection of the functions that are described below. 

Density plot from raw IMS data
This function generates a density plot of one raw “MCC-IMS BioScout” file. 
Select the input file via the file’s dialog window that opens when executing the command. 

Peak detection
Run the peak identification PEAX on a set of raw “MCC-IMS BioScout” files. 

Proceed to the following steps respectively using the windows that pop up when executing the command:

1- Select a folder containing the raw “MCC-IMS BioScout” files on which you want to perform the peak detection. WARNING: the folder must contain only raw “MCC-IMS BioScout”.

2-Select an empty folder to save the peak list resulting from the peak detection, the name of the files will be identical to the corresponding name of the raw file. 

3-Select a PEAX parameter file (.cfg)


Peak Alignment
This functionality generates a CSV file containing a set of label coordinates “r” and “t” corresponding to the alignment of the peak lists. 

Bellow are described the steps that should be used while running the command:

1- Select a folder containing the raw “MCC-IMS BioScout” files on which you want to perform the peak detection. WARNING: the folder must contain only raw “MCC-IMS BioScout”.

2- Select a folder containing the peak list files on which you want to perform the peak alignment. WARNING: the folder must contain only CSV peak list file resulting from the peak detection function. The number of files and their names should be identical to the one in the raw data folder.

3- Input a distance threshold (must be an integer). This distance corresponds to the maximum Euclidean distance for two peaks to be aligned. The recommended value is 5.

At the end of the alignment the user will be asked if he/she wants to print a plot summarizing the alignment. It could be done by typing “Y”. To finalize the alignment close the plot windows and enter the name of the file in which you want to save the peak alignment. 

n.b : Please keep in mind that if the number of peak lists to be aligned is high and the distance’s threshold is low, the computation time might be increased drastically.


Train Random forest classifier and predict
When executing this option a random forest will be trained and will predict the selected data.

Follow this steps to run the command:
Select a directory containing training peak lists.WARNING: the folder must contain only CSV peak list file resulting of the peak detection function. The number of the files and their names should be identical to the one in the raw data folder.
Select the peak alignment file.
Select the class labels for the train data.
Select the directory containing testing peak lists.
Input the maximum detection distance (must be an integer). This threshold corresponds to the maximum Euclidean distance between two peaks to be included in the indicator matrix used to train the Random Forest classifier.

Train MLP classifier and predict
This method uses a multilayer perceptron classifier to predict the class of the peak lists. 

Below are described the steps that should be used while running the command:

1-Select a folder containing a set of peak lists corresponding to the training dataset. WARNING: the folder must contain only CSV peak list file resulting of the peak detection function. The number of the files and their names should be identical to the one in the raw data folder.

2-select the class-label file corresponding to the set of peak list selected.

3-Select a peak alignment file that has been created by the peak alignment function. 

4-Input the maximum detection distance (must be an integer). This threshold corresponds to the maximum Euclidean distance between two peaks to be included in the intensity matrix used to train the MLP classifier. 

You will be asked if you want to perform a 5 fold-cross validation on the training dataset, input “Y” if you want to see the accuracy of the model given for the training dataset. 

After training the MLP classifier with the complete training dataset, select a folder of the peak lists you want to predict using this classifier. 
Enter the name of the file to save results from the prediction made by MLP classifier. 


