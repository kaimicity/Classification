
# Calculate the accuracy of the prediction
def accuacy(y_hat, y):
    T = 0.000
    for i in range(len(y_hat)):
        if y[i].any() == y_hat[i].any():
            T += 1.000
    return T / float(len(y_hat))


# Transform the output to the origin format and export it
def export(predictions):
    res = []
    predicted = False
    for pre in predictions:
        for i in range(len(pre)):
            if pre[i] == 1:
                res.append(i)
                predicted = True
        if predicted == False:
            res.append(-1)
        predicted = False
    return res
