import numpy as np


def knn(train_x, train_y, test_x, k=1):
    res = []
    for tst in test_x:
        distances = []
        vote_0 = 0
        vote_1 = 0
        vote_2 = 0
        vote_3 = 0
        vote_4 = 0
        for x in train_x:
            distance = 0
            for i in range(len(x)):
                # print(tst.dtype)
                # print(x.dtype)
                d = tst[i] - x[i]
                d = d ** 2
                distance += d
            distances.append(distance)
        distances_index = np.argsort(distances)
        for i in range(k):
            if i < len(distances):
               obs = distances_index[i]
               obs = train_y[obs]
               if obs == 0:
                    vote_0 += 1
               elif obs == 1:
                    vote_1 +=1
               elif obs == 2:
                    vote_2 += 1
               elif obs == 3:
                    vote_3 +=1
               elif obs == 4:
                    vote_4 += 1
        votes = [vote_0, vote_1, vote_2, vote_3, vote_4]
        votes_index = np.argsort(votes)
        res.append(votes_index[4])
    return res


def accuracy(y_hat, y):
    T = 0.000
    for i in range(len(y_hat)):
        if y[i] == y_hat[i]:
            T += 1.000
    return T / float(len(y_hat))
