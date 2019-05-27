from preparation import sig_mean_train, sig_mean_vali, sig_max_train, sig_max_vali, sig_min_train, sig_min_vali, x_test_mean
import numpy as np
import util
import pandas as pd
import time

# Transform data from dataframes to more executable arrays
sig_mean_train = sig_mean_train.values
sig_mean_vali = sig_mean_vali.values
sig_min_train = sig_min_train.values
sig_min_vali = sig_min_vali.values
sig_max_train = sig_max_train.values
sig_max_vali = sig_max_vali.values
x_test_mean = x_test_mean.values

# Extract input set and output set respectively
mean_train_x = sig_mean_train[:, : (len(sig_mean_train[0]) - 1)]
mean_train_y = sig_mean_train[:, (len(sig_mean_train[0]) - 1)]
mean_vali_x = sig_mean_vali[:, : (len(sig_mean_vali[0]) - 1)]
mean_vali_y = sig_mean_vali[:, (len(sig_mean_vali[0]) - 1)]
min_train_x = sig_min_train[:, : (len(sig_min_train[0]) - 1)]
min_train_y = sig_min_train[:, (len(sig_min_train[0]) - 1)]
min_vali_x = sig_min_vali[:, : (len(sig_min_vali[0]) - 1)]
min_vali_y = sig_min_vali[:, (len(sig_min_vali[0]) - 1)]
max_train_x = sig_max_train[:, : (len(sig_max_train[0]) - 1)]
max_train_y = sig_max_train[:, (len(sig_max_train[0]) - 1)]
max_vali_x = sig_max_vali[:, : (len(sig_max_vali[0]) - 1)]
max_vali_y = sig_max_vali[:, (len(sig_max_vali[0]) - 1)]

vali_mean_res = util.knn(mean_train_x, mean_train_y, mean_vali_x)
print(util.accuracy(mean_vali_y, vali_mean_res))

vali_min_res = util.knn(min_train_x, min_train_y, min_vali_x)
print(util.accuracy(min_vali_y, vali_min_res))

vali_max_res = util.knn(max_train_x, max_train_y, max_vali_x)
print(util.accuracy(max_vali_y, vali_max_res))

res = util.knn(mean_train_x, mean_train_y, x_test_mean, 2)
classes = []
for id in res:
    if id == 0:
        classes.append('air')
    elif id == 1:
        classes.append('book')
    elif id == 2:
        classes.append('hand')
    elif id == 3:
        classes.append('knife')
    elif id == 4:
        classes.append('plastic case')
final_res = pd.DataFrame({'Class ID': res, 'Class': classes})
final_res.to_csv('PredictedClasses.csv')