from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()


def data_standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    data_standard = np.zeros(digits.data.shape)
    for i in range(digits.data.shape[0]):
        for j in range(digits.data.shape[1]):
            if sigma[j] != 0:
                data_standard[i][j] = (data[i][j] - mu[j]) / sigma[j]
            else:
                data_standard[i][j] = data[i][j]

    return data_standard


def target_in_one_hot_encoding(target):
    target_one_hot_encoding = np.zeros((digits.data.shape[0], 10))
    for i in range(digits.data.shape[0]):
        number = target[i]
        target_one_hot_encoding[i][number] = 1

    return target_one_hot_encoding


def initialization_w():
    w_init = sigma_for_initialization * np.random.randn(10, digits.data.shape[1])
    return w_init


def initialization_b():
    b_init = sigma_for_initialization * np.random.randn(10)
    return b_init


def y_train_calculation(w, bias):
    y_train = np.zeros((data_train.shape[0], 10))
    for i in range(data_train.shape[0]):
        y_train[i] = softmax(np.dot(w, data_train[i]) + bias)
    return y_train


def y_valid_calculation(w, bias):
    y_valid = np.zeros((data_valid.shape[0], 10))
    for i in range(data_valid.shape[0]):
        y_valid[i] = softmax(np.dot(w, data_valid[i]) + bias)
    return y_valid


def softmax(u):
    maximum = np.amax(u)
    u -= maximum
    y_i = np.float64(np.exp(u))/(np.sum(np.float64(np.exp(u))))
    return y_i


def get_error(data, target, w, bias):
    err = 0
    for i in range(data.shape[0]):
        y_i = softmax(np.dot(w, data[i]) + bias)
        err += np.dot(target[i], np.log(y_i))

    return -err


# def get_error2(data, target):
#     err = 0
#     for i in range(data.shape[0]):
#         a = np.dot(W, data[i]) + b
#         p = np.amax(a)
#         a = a - p
#         for k in range(target.shape[1]):
#             err += target[i][k] * (a[k] - math.log(np.sum(np.float64(np.exp(a)))))
#
#     return -err


def nabla_ew_calculation(y_train):
    nabla_ew = np.dot((y_train - target_train).T, data_train)
    return nabla_ew


def nabla_eb_calculation(y_train):
    um = np.ones(data_train.shape[0])
    nabla_eb = np.dot((y_train - target_train).T, um)
    return nabla_eb


def accuracy_calculation(y_matrix, target):

    confusion_matrix = np.zeros((10, 10))

    for k in range(y_matrix.shape[0]):
        i = np.argmax(target[k])
        j = np.argmax(y_matrix[k])
        confusion_matrix[i][j] += 1

    temp = 0

    for i in range(confusion_matrix.shape[0]):
        temp += confusion_matrix[i][i]

    return temp / y_matrix.shape[0]


def gradient_descent():

    w = initialization_w()
    bias = initialization_b()

    e_valid = 0
    e_valid_new = 0
    y_train = 0
    y_valid = 0

    accuracy_list_valid = []
    accuracy_list_train = []
    e_valid_list = []
    e_train_list = []

    counter = 0

    while e_valid >= e_valid_new:

        e_valid = get_error(data_valid, target_valid, w, bias)

        y_train = y_train_calculation(w, bias)
        y_valid = y_valid_calculation(w, bias)

        e_valid_list.append(e_valid)
        acc_train = accuracy_calculation(y_train, target_train)
        acc_valid = accuracy_calculation(y_valid, target_valid)
        accuracy_list_train.append(acc_train)
        accuracy_list_valid.append(acc_valid)

        w_temp = w - gamma * nabla_ew_calculation(y_train)
        b_temp = bias - gamma * nabla_eb_calculation(y_train)

        e_train = get_error(data_train, target_train, w, bias)
        e_train_new = get_error(data_train, target_train, w_temp, b_temp)

        e_train_list.append(e_train)

        if counter % 5 == 0:
            print("iteration [{0}] :\nE_train ={1}\nE_valid ={2}\nAccuracy train={3}\nAccuracy valid={4}\n "
                  .format(counter, e_train, e_valid, acc_train, acc_valid))

        if e_train_new < e_train:
            w = w_temp
            bias = b_temp

        e_valid_new = get_error(data_valid, target_valid, w, bias)
        counter += 1

    accuracy_valid = accuracy_calculation(y_valid, target_valid)

    draw_graphics(accuracy_list_train, accuracy_list_valid, e_train_list, e_valid_list)

    return accuracy_valid


def draw_graphics(acc_train, acc_valid, e_train, e_valid):

    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 2, 1)
    ax_2 = fig.add_subplot(2, 2, 2)
    ax_3 = fig.add_subplot(2, 2, 3)
    ax_4 = fig.add_subplot(2, 2, 4)

    ax_1.plot(list(range(len(acc_train))), acc_train)
    ax_1.set_title("Accuracy на train")
    ax_2.plot(list(range(len(e_train))), e_train)
    ax_2.set_title("Ошибка на train")
    ax_3.plot(list(range(len(acc_valid))), acc_valid, color='red')
    ax_3.set_title("Accuracy на valid")
    ax_4.plot(list(range(len(e_valid))), e_valid, color='red')
    ax_4.set_title("Ошибка на valid")


N = digits.data.shape[0]

gamma = 0.005
sigma_for_initialization = 0.05

ind = np.arange(N)
np.random.shuffle(ind)

ind_train = ind[:np.int32(0.8*len(ind))]
ind_valid = ind[np.int32(0.8*len(ind)):np.int32(len(ind))]

data_st = data_standardization(digits.data)
target_enc = target_in_one_hot_encoding(digits.target)

data_train = data_st[ind_train]
target_train = target_enc[ind_train]

data_valid = data_st[ind_valid]
target_valid = target_enc[ind_valid]

# W = initialization_w()
# b = initialization_b()
# print(get_error(data_train, target_train, W, b))
# print(get_error(data_valid, target_valid, W, b))

accuracy_final = gradient_descent()
print("Final accuracy =", accuracy_final)

plt.show()