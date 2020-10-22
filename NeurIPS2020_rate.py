import numpy as np
import matplotlib.pyplot as plt


def kernel_matrix(x, y=None, sigma=None):
    if y is None:
        y = x
    if len(x.shape) == 1:
        x = np.reshape(x, [-1, 1])
    if len(y.shape) == 1:
        y = np.reshape(y, [-1, 1])
    x_squared = np.sum(np.power(x, 2), axis=-1, keepdims=True)
    y_squared = np.sum(np.power(y, 2), axis=-1, keepdims=True).T
    xy_inner = np.matmul(x, y.T)
    kernel_input = x_squared + y_squared - 2 * xy_inner
    return np.exp(-0.5 * sigma * kernel_input)


def x_function(q, z, n):
    if q == 0:
        return np.exp(-0.5 * np.power(z, 2)) * np.sin(2 * z) + 0.3 * np.random.normal(size=n)
    elif q == 1:
        return np.exp(-0.5 * np.power(z, 2)) * np.sin(2 * z) + 3 * np.random.normal(size=n)
    else:
        return z + 0.3 * np.random.normal(size=n)


def emp_err(n_int, n_com, z_int, x_int, z_com, x_com, sigma_z, sigma_x, reg):
    kk_z = kernel_matrix(z_int, z_com, sigma_z)
    kk_x = kernel_matrix(x_com, x_int, sigma_x)
    k_x = kernel_matrix(x_com, sigma=sigma_x)
    k_z = kernel_matrix(z_com, sigma=sigma_z)
    w = np.linalg.inv(k_z + reg * np.identity(n_com))
    second = (1 / n_int) * np.trace(kk_z @ w @ kk_x)
    third = (1 / n_int) * np.trace(kk_z @ w @ k_x @ w @ kk_z.T)
    return 1 - 2 * second + third


# Set the hyper-parameters.
np.random.seed(22)
N_int = 1000
N_com = [50, 100, 300, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
N_rep = 1000
sigma_X = 0.1
sigma_Z = 0.1
# lamb = np.power(N_com, -0.75) / 1000
lamb = 1 / (10000000 * np.power(np.asarray(N_com), 0.25))
Q = 3

Z_int = [np.random.normal(size=N_int) for q in range(Q)]
X_int = [x_function(q, Z_int[q], N_int) for q in range(Q)]


def for_each_z_second(n_rep, q, z, x, sigma_x):
    x_prime = x_function(q, z, n_rep)
    return np.mean(np.exp(-0.5 * sigma_x * np.power(x - x_prime, 2)))


def for_each_z_third(n_rep, q, z, sigma_x):
    x = x_function(q, z, n_rep)
    k_x = kernel_matrix(x, sigma=sigma_x)
    return (1 / (n_rep * (n_rep - 1))) * (np.sum(k_x) - np.trace(k_x))


Second = [np.mean(np.asarray([for_each_z_second(N_rep, q, Z_int[q][p], X_int[q][p], sigma_X)
                              for p in range(N_int)])) for q in range(Q)]
Third = [np.mean(np.asarray([for_each_z_third(N_rep, q, p, sigma_X) for p in Z_int[q]])) for q in range(Q)]
error_true = [1 - 2 * Second[q] + Third[q] for q in range(Q)]
error_emp = [np.zeros(len(N_com)) for q in range(Q)]
for q in range(Q):
    for i in range(len(N_com)):
        Z_com = np.random.normal(size=N_com[i])
        X_com = x_function(q, Z_com, N_com[i])
        error_emp[q][i] = emp_err(N_int, N_com[i], Z_int[q], X_int[q], Z_com, X_com, sigma_Z, sigma_X, lamb[i])
print(error_true, error_emp)

fig, axes = plt.subplots(1, 3)
titles = ["(a)", "(b)", "(c)"]
for q in range(Q):
    axes[q].plot(N_com, error_emp[q], linestyle="-", marker="o", label="Risk of Empirical Estimate")
    axes[q].hlines(error_true[q], xmin=0, xmax=N_com[-1], linestyle="--", color="red", label="Risk of True Function")
    axes[q].set_xlabel("No. of samples")
    axes[q].set_title(titles[q])
    # axes[q].legend()
# plt.figure(2)
# plt.scatter(Z_int, X_int)
plt.show()
