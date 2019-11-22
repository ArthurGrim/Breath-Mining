
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd
from tkinter.filedialog import askopenfilename


file = askopenfilename()

data = pd.read_csv(file,low_memory=False, comment='#')

print(data.head())

data = data.drop(data.index[[0]])
data = data.drop(data.columns[[0,1]], axis=1)
print(data.head())


# plot heatmap
ax = sns.heatmap(data.T)

# turn the axis label
for item in ax.get_yticklabels():
    item.set_rotation(0)

for item in ax.get_xticklabels():
    item.set_rotation(90)

# save figure
plt.savefig('seabornPandas.png', dpi=100)
plt.show()
