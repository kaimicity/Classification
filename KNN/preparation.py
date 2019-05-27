import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load training data and test data
x = pd.read_csv('X.csv')
y = pd.read_csv('Y.csv')
test_x = pd.read_csv('XToClassify.csv')

# Explore Missing Values
# print(x.isnull().values.any())
# print(test_x.isnull().values.any())

# Add column names
columns = []
for i in range(256):
    columns.append("mean" + str(i))
for i in range(256):
    columns.append("min" + str(i))
for i in range(256):
    columns.append("max" + str(i))
x.columns = columns
y.columns = ['Y']
test_x.columns = columns
# print(x.describe())

# Normalize input data
x_norm = (x - x.mean(axis=0)) / (x.max(axis=0) - x.min(axis=0))
x_test_norm = (test_x - x.mean(axis=0)) / (x.max(axis=0) - x.min(axis=0))
x_test_mean = x_test_norm[columns[0:256]]
# x_test_mean.info()

# Merge input and output to accord data for splitting
sig_mean = pd.merge(x_norm[columns[0:256]], y, left_index=True, right_index=True)
# sig_mean.info()
sig_min = pd.merge(x_norm[columns[256:512]], y, left_index=True, right_index=True)
# sig_min.info()
sig_max = pd.merge(x_norm[columns[512:768]], y, left_index=True, right_index=True)
# sig_max.info()

# Split training data into training set and validation set
sig_mean_train, sig_mean_vali = train_test_split(sig_mean, test_size=0.2, random_state=0)
sig_min_train, sig_min_vali = train_test_split(sig_min, test_size=0.2, random_state=0)
sig_max_train, sig_max_vali = train_test_split(sig_max, test_size=0.2, random_state=0)