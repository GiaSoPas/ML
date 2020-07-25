import numpy as np
import matplotlib.pyplot as plt

# вычисление числа пи
def calculationpi(n):

    n_in = 0
    del x_in[:]
    del y_in[:]
    del x_out[:]
    del y_out[:]

    for i in range(0, n):
        if ((xy[i][0] - 0.5) ** 2 + (xy[i][1] - 0.5) ** 2) < 0.5**2:
            x_in.append(xy[i][0])
            y_in.append(xy[i][1])
            n_in += 1
        else:
            x_out.append(xy[i][0])
            y_out.append(xy[i][1])

    return n_in / n * 4

# строит график зависимости
def depend():
    M = np.arange(10, 10000, 20)
    pi = []

    for i in range(0, M.size):
        pi.append(calculationpi(M[i]))

    ax2.plot(M, pi)

# рисует единичный квадрат и окружность с точками
def draw():

    ax1.scatter(x_in, y_in, color='red', s=1)
    ax1.scatter(x_out, y_out, color='blue', s=1)
    circ = plt.Circle((0.5, 0.5), 0.5, fill=False)
    pol = plt.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)], fill=False)
    ax1.add_patch(circ)
    ax1.add_patch(pol)


N = 10000
xy = np.random.rand(N, 2)
x_in = []
y_in = []
x_out = []
y_out = []
print(calculationpi(1000))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

depend()
draw()

plt.show()


