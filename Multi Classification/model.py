from preparation import sig_mean_train, sig_mean_vali, sig_max_train, sig_max_vali, sig_min_train, sig_min_vali, x_test_mean
import numpy as np
import time
import pandas as pd
import util
from sklearn.neural_network import MLPClassifier

# Transform data from dataframes to more executable arrays
sig_mean_train = sig_mean_train.values
sig_mean_vali = sig_mean_vali.values
sig_min_train = sig_min_train.values
sig_min_vali = sig_min_vali.values
sig_max_train = sig_max_train.values
sig_max_vali = sig_max_vali.values
x_test_mean = x_test_mean.values

# Extract input set and output set respectively
mean_train_x = sig_mean_train[:, : (len(sig_mean_train[0]) - 5)]
mean_train_y = sig_mean_train[:, len(sig_mean_train[0]) - 5:]
mean_vali_x = sig_mean_vali[:, : (len(sig_mean_vali[0]) - 5)]
mean_vali_y = sig_mean_vali[:, len(sig_mean_vali[0]) - 5:]
min_train_x = sig_min_train[:, : (len(sig_min_train[0]) - 5)]
min_train_y = sig_min_train[:, len(sig_min_train[0]) - 5:]
min_vali_x = sig_min_vali[:, : (len(sig_min_vali[0]) - 5)]
min_vali_y = sig_min_vali[:, len(sig_min_vali[0]) - 5:]
max_train_x = sig_max_train[:, : (len(sig_max_train[0]) - 5)]
max_train_y = sig_max_train[:, len(sig_max_train[0]) - 5:]
max_vali_x = sig_max_vali[:, : (len(sig_max_vali[0]) - 5)]
max_vali_y = sig_max_vali[:, len(sig_max_vali[0]) - 5:]

# Create a multi-layer perceptron classifier
nn_classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=28, activation='tanh')

# Use mean, min, max values of signals to train models and evaluate their performance
avg_acc_mean, avg_elapsed_time_mean, avg_loss_mean = [], [], []
avg_acc_min, avg_elapsed_time_min, avg_loss_min = [], [], []
avg_acc_max, avg_elapsed_time_max, avg_loss_max = [], [], []
for i in range(5):
    start = time.time()
    nn_classifier.fit(mean_train_x, mean_train_y)
    end = time.time()
    acc_mean = util.accuacy(nn_classifier.predict(mean_vali_x), mean_vali_y)
    avg_loss_mean.append(nn_classifier.loss_)
    avg_acc_mean.append(acc_mean)
    avg_elapsed_time_mean.append(end - start)

    start = time.time()
    nn_classifier.fit(min_train_x, min_train_y)
    end = time.time()
    acc_min = util.accuacy(nn_classifier.predict(min_vali_x), min_vali_y)
    avg_loss_min.append(nn_classifier.loss_)
    avg_acc_min.append(acc_min)
    avg_elapsed_time_min.append(end - start)

    start = time.time()
    nn_classifier.fit(max_train_x, max_train_y)
    end = time.time()
    acc_max = util.accuacy(nn_classifier.predict(max_vali_x), max_vali_y)
    avg_loss_max.append(nn_classifier.loss_)
    avg_acc_max.append(acc_max)
    avg_elapsed_time_max.append(end - start)

print(np.mean(avg_loss_mean))
print(np.mean(avg_acc_mean))
print(np.mean(avg_elapsed_time_mean))
print(np.mean(avg_loss_min))
print(np.mean(avg_acc_min))
print(np.mean(avg_elapsed_time_min))
print(np.mean(avg_loss_max))
print(np.mean(avg_acc_max))
print(np.mean(avg_elapsed_time_max))

# Predict for the test dataset and interpret and export the result
res = util.export(nn_classifier.predict(x_test_mean))
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
    elif id == -1:
        classes.append('can\'t idntify')
final_res = pd.DataFrame({'Class ID': res, 'Class': classes})
final_res.to_csv('PredictedClasses.csv')


