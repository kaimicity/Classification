from preparation import sig_mean_train, sig_mean_vali, sig_max_train, sig_max_vali, sig_min_train, sig_min_vali, x_test_mean
import numpy as np
import util
import time

# Transform data from dataframes to more executable arrays
sig_mean_train = sig_mean_train.values
sig_mean_vali = sig_mean_vali.values
sig_min_train = sig_min_train.values
sig_min_vali = sig_min_vali.values
sig_max_train = sig_max_train.values
sig_max_vali = sig_max_vali.values

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

# Add x0 to the input
mean_train_x = np.c_[np.ones_like(mean_train_x[:, 0]), mean_train_x]
mean_vali_x = np.c_[np.ones_like(mean_vali_x[:, 0]), mean_vali_x]
min_train_x = np.c_[np.ones_like(min_train_x[:, 0]), min_train_x]
min_vali_x = np.c_[np.ones_like(min_vali_x[:, 0]), min_vali_x]
max_train_x = np.c_[np.ones_like(max_train_x[:, 0]), max_train_x]
max_vali_x = np.c_[np.ones_like(max_vali_x[:, 0]), max_vali_x]

# All theta values are initially set to 0
init_theta = []
for i in range(len(max_train_x[0])):
    init_theta.append(0)
init_theta = np.array(init_theta)

# Use mean, min, max values of signals to train models and evaluate their performance
start = time.time()
mean_alpha = 2.7
mean_lamda = 0
mean_train_theta, mean_train_loss = util.gradient_decent(mean_train_x, init_theta, mean_train_y, mean_alpha, mean_lamda)
print(mean_train_loss)
mean_vali_acc = util.accuracy(mean_vali_y, util.model_func(mean_vali_x, mean_train_theta))
print(mean_vali_acc)
end = time.time()
print(end - start)

start = time.time()
min_alpha = 4.1
min_lamda = 0
min_train_theta, min_train_loss = util.gradient_decent(min_train_x, init_theta, min_train_y, min_alpha, min_lamda)
print(min_train_loss)
min_vali_acc = util.accuracy(min_vali_y, util.model_func(min_vali_x, min_train_theta))
print(min_vali_acc)
end = time.time()
print(end - start)

start = time.time()
max_alpha = 3.3
max_lamda = 0
max_train_theta, max_train_loss = util.gradient_decent(max_train_x, init_theta, max_train_y, max_alpha, max_lamda)
print(max_train_loss)
max_vali_acc = util.accuracy(max_vali_y, util.model_func(max_vali_x, max_train_theta))
print(max_vali_acc)
end = time.time()
print(end - start)

# Predict for test dataset
x_test_mean = x_test_mean.values
x_test_mean = np.c_[np.ones_like(x_test_mean[:, 0]), x_test_mean]
prediction = util.predict(x_test_mean, mean_train_theta)
print(prediction)