import numpy as np
import matplotlib.pyplot as plt



def regression(M):

    design_nmatrix = np.zeros((N, M))

    for i in range(M):
        design_nmatrix[:, i] = x ** i

    w = np.dot(np.dot(np.linalg.inv(np.dot(design_nmatrix.T, design_nmatrix)), design_nmatrix.T), t)

    y = np.dot(design_nmatrix, w)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(x, z, color='k')
    ax1.scatter(x, t, s=1)
    ax1.plot(x, y, color='r')

    plt.show()


def fault_e():

    E = []
    for i in range(1, 100):

        design_nmatrix = np.zeros((N, i))

        for j in range(i):
            design_nmatrix[:, j] = x ** j

        w = np.dot(np.dot(np.linalg.inv(np.dot(design_nmatrix.T, design_nmatrix)), design_nmatrix.T), t)
        y = np.dot(design_nmatrix, w)
        r = 0

        for k in range(N):
            r += 0.5*(y[k]-t[k])**2

        E.append(r)

    m = np.arange(1, 100, 1)
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(m,E)
    plt.show()


N = 1000

x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

regression(2)
regression(8)
regression(100)

fault_e()

