import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd
from tkinter.filedialog import askopenfilename


def main(file):

    print("\n\n -=Generating Density Plot=- \n\n" )

    data = pd.read_csv(file,low_memory=False, comment='#')

    data = data.drop(data.index[[0]])
    data = data.drop(data.columns[[0,1]], axis=1)


    # plot heatmap
    ax = sns.heatmap(data.T)

    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    # save figure
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1])
