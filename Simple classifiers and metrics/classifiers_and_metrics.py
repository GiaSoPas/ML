import numpy as np
import matplotlib.pyplot as plt


def classifier(t):

    for i in range(N):
        if arr_basketball[i] >= t:
            train_label_basket[i] = 1
        else:
            train_label_basket[i] = 0
        if arr_football[i] >= t:
            train_label_foot[i] = 1
        else:
            train_label_foot[i] = 0


def true_positives():

    tp = np.sum(train_label_basket)
    return tp


def true_negatives():

    tn = 1000 - np.sum(train_label_foot)
    return tn


def false_positives():

    fp = np.sum(train_label_foot)
    return fp


def false_negatives():

    fn = 1000 - np.sum(train_label_basket)
    return fn


def accuracy_calculation():

    accuracy = (true_positives() + true_negatives()) / (2*N)
    return accuracy


def precision_calculation():

    if (true_negatives() + false_positives()) == 0:
        return 1
    else:
        precision = true_positives() / (true_positives() + false_positives())
        return precision


def recall_calculation():

    recall = true_positives() / (true_positives() + false_negatives())
    return recall


def f1_score_calculation():

    f1_score = 2*((precision_calculation()*recall_calculation())/(precision_calculation() + recall_calculation()))
    return f1_score


def alpha_calculation():

    alpha = false_positives() / (false_positives() + true_negatives())
    return alpha


def beta_calculation():

    beta = false_negatives() / (false_negatives() + true_positives())
    return beta


def building_roc():

    best_accuracy = 0
    best_t = 0
    alpha_list = []
    recall_list = []
    for t in range(1, 300, 1):
        classifier(t)
        alpha = alpha_calculation()
        recall = recall_calculation()
        alpha_list.append(alpha)
        recall_list.append(recall)
        ne_best_accuracy = accuracy_calculation()

        if ne_best_accuracy > best_accuracy:
            best_accuracy = ne_best_accuracy
            best_t = t

    best_accuracy_and_metrics(best_t)

    s = auc_calculation(alpha_list, recall_list)
    print("AUC =", s)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(alpha_list, recall_list, color='r')
    ax1.plot([alpha_list[len(alpha_list)-1], recall_list[0]], [recall_list[len(recall_list)-1], alpha_list[0]], "s--")
    plt.show()


def auc_calculation(alpha_list, recall_list):

    s = 0
    for i in range(len(alpha_list)-1):
        h = abs(alpha_list[i] - alpha_list[i+1])
        s += (h*(recall_list[i] + recall_list[i+1]))/2

    return s


def best_accuracy_and_metrics(t):
    print("Best t =", t)
    classifier(t)
    print("TP =", true_positives(), "TN =", true_negatives(), "FN =", false_negatives(), "FP =", false_positives())
    print("Accuracy =", accuracy_calculation())
    print("Precision =", precision_calculation())
    print("Recall =", recall_calculation())
    print("F1-SCORE =", f1_score_calculation())
    print("ALPHA =", alpha_calculation(), "BETA =", beta_calculation())


N = 1000
mu_1 = 193
sigma_1 = 7
mu_0 = 180
sigma_0 = 6

arr_basketball = sigma_1*np.random.randn(N) + mu_1
arr_football = sigma_0*np.random.randn(N) + mu_0

label_basket = np.ones(N, dtype=int)
label_foot = np.zeros(N, dtype=int)

train_label_basket = np.ones(N, dtype=int)
train_label_foot = np.zeros(N, dtype=int)

building_roc()