from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def initialization_centroids():
    d_vec = np.random.uniform(-3, 3, 2)
    return d_vec


def distance(a, b):
    return np.linalg.norm(a-b, axis=1)


centers = [[-1, -1], [0, 1], [1, -1]]
X, _ = make_blobs(n_samples=3000, centers=centers, cluster_std=0.5)

E = np.zeros(10)
final_centroids = np.zeros((10, 10, 2))

n = 10

for k in range(1, 11):

    centroids = np.zeros((k, 2))
    prev_centroids = np.zeros((k, 2))

    for i in range(k):
        centroids[i] = initialization_centroids()

    for iteration in range(n):

        sum_in_cluster = np.zeros((k, 2))
        number_in_cluster = np.zeros(k)
        E_temp = 0

        for index in range(X.shape[0]):
            distances = distance(centroids, X[index])
            sum_in_cluster[np.argmin(distances)] += X[index]
            number_in_cluster[np.argmin(distances)] += 1
            E_temp += (np.amin(distances))**2

        prev_centroids = centroids.copy()

        for ind in range(k):
            if number_in_cluster[ind] != 0:
                centroids[ind] = sum_in_cluster[ind] / number_in_cluster[ind]

        if np.array_equal(centroids, prev_centroids) or iteration == n-1:
            E[k-1] = E_temp
            for t in range(k):
                final_centroids[k-1][t] = centroids[t]
            break

print(E)

D = np.zeros(9)
D[0] = 5
for i in range(1, 9):
    D[i] = abs((E[i] - E[i+1]))/abs((E[i-1] - E[i]))

print(D)

optimal_clusters = np.argmin(D) + 1

fig = plt.figure()
ax_1 = fig.add_subplot(2, 2, 1)
ax_1.plot(list(range(1, 11)), E)
ax_2 = fig.add_subplot(2, 2, 2)
ax_2.plot(list(range(2, 10)), D[1:])
ax_3 = fig.add_subplot(2, 2, 3)

colors = []

for ind in range(X.shape[0]):
    distances = distance(final_centroids[optimal_clusters - 1][:optimal_clusters], X[ind])
    colors.append(np.argmin(distances))

ax_3.scatter(X[:, 0], X[:, 1], s=1, c=colors)
ax_3.scatter(final_centroids[optimal_clusters - 1][:optimal_clusters, 0], final_centroids[optimal_clusters - 1][:optimal_clusters, 1], 25, c='r', marker="X")

plt.show()








