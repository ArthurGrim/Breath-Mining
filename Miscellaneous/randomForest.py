import pandas as pd
import numpy as np

dataset = pd.read_csv('indicatorMatrix.csv')


labels = dataset.iloc[:,-1]
dataset = dataset.iloc[:,1:-2]



print(dataset.head())
print(labels.head())


from sklearn.ensemble import RandomForestRegressor




regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
