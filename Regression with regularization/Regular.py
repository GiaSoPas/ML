import numpy as np
import matplotlib.pyplot as plt


def get_error(w_c, x_v, t_v, u_c):

    design_nmatrix = create_design_matrix(x_v, u_c)
    y = np.dot(design_nmatrix, w_c)
    e_c = 0
    for k in range(len(x_v)):
        e_c += 0.5*(y[k]-t_v[k])**2

    return e_c


def create_design_matrix(x_c, u_c):
    design_nmatrix = np.zeros((len(x_c), len(u_c) + 1))
    design_nmatrix[:, 0] = 1

    for index in range(len(u_c)):
        design_nmatrix[:, index + 1] = u_c[index](x_c)

    return design_nmatrix


def get_param(u_c, l_cur, x_c, t_c):
    design_nmatrix = create_design_matrix(x_c, u_c)
    w = np.dot(np.dot(np.linalg.inv(np.dot(design_nmatrix.T, design_nmatrix)+np.dot(l_cur, np.eye(len(u_c) + 1))), design_nmatrix.T), t_c)
    return w


def func(u_c):
    g = lambda p: p**u_c
    g.__name__ = "polinom degree " + str(u_c)
    return g


N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

ind = np.arange(N)
np.random.shuffle(ind)

ind_train = ind[: np.int32(0.8*len(ind))]
ind_valid = ind[np.int32(0.8*len(ind)):np.int32(0.9*len(ind))]
ind_test = ind[np.int32(0.9*len(ind)):np.int32(len(ind))]

x_train = x[ind_train]
t_train = t[ind_train]
x_valid = x[ind_valid]
t_valid = t[ind_valid]
x_test = x[ind_test]
t_test = t[ind_test]

# t_test x_test  x_valid t_valid
inter_num = 1000

Em = 10**10

lam = np.array([0.000000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 10000, 100000, 1000000])
u = [np.sin, np.cos, np.exp, np.sqrt, func(2), func(3), func(4), func(5), func(6), func(7), func(8), func(9), func(10)]
best_u = 0
best_lam = 0
best_w = 0

for i in range(inter_num):
    u_cur = np.random.choice(u, np.random.randint(0, len(u)), replace=False)  # функция даей случайную функця, или команада choice де флаг без повторяения np.random.choice() или get_current_num_funcs(u)
    lam_cur = np.random.choice(lam)  # либо np.random.choice[lam] или np.random.randint()
    w_cur = get_param(u_cur, lam_cur, x_train, t_train)
    E_cur = get_error(w_cur, x_valid, t_valid, u_cur)

    if E_cur < Em:
        Em = E_cur
        best_u = u_cur
        best_lam = lam_cur
        best_w = w_cur

E_test = get_error(best_w, x_test, t_test, best_u)
print(Em)
print(E_test)
print(best_u)
print(best_lam)


for i in range(len(best_u)):
    print(best_u[i].__name__)


des_m = create_design_matrix(x, best_u)
y_best = np.dot(des_m, best_w)
#
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#
ax1.plot(x, z, color='k')
ax1.scatter(x, t, s=1)
ax1.plot(x, y_best, color='r')
#
plt.show()