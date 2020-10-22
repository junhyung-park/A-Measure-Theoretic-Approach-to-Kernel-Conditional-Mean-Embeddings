import numpy as np
import matplotlib.pyplot as plt


# The following function computes the kernel matrix using the Gaussian kernel between vectors x and y, with bandwidth
# sigma. If y=None, the Gram matrix is computed based on x.
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


# The following function evaluates the function M^2 conditioned on arg (see Section 5.1)
def mcmd(arg, k_x, k_x_prime, k_x_x_prime, z, z_prime, w, w_prime, sigma_z):
    kk_z = kernel_matrix(z, np.reshape(arg, [1, -1]), sigma_z)
    kk_z_prime = kernel_matrix(z_prime, np.reshape(arg, [1, -1]), sigma_z)
    first = kk_z.T @ w @ k_x @ w @ kk_z
    second = kk_z.T @ w @ k_x_x_prime @ w_prime @ kk_z_prime
    third = kk_z_prime.T @ w_prime @ k_x_prime @ w_prime @ kk_z_prime
    return (first - 2 * second + third)[0, 0]


# The following function evaluates the (unnormalised) witness function at arg_x conditioned on arg_z (see Section 5.1)
def witness(arg_z, arg_x, x, x_prime, z, z_prime, w, w_prime, sigma_x, sigma_z):
    kk_x = kernel_matrix(x, np.reshape(arg_x, [1, -1]), sigma_x)
    kk_x_prime = kernel_matrix(x_prime, np.reshape(arg_x, [1, -1]), sigma_x)
    kk_z = kernel_matrix(z, np.reshape(arg_z, [1, -1]), sigma_z)
    kk_z_prime = kernel_matrix(z_prime, np.reshape(arg_z, [1, -1]), sigma_z)
    return (kk_z.T @ w @ kk_x - kk_z_prime.T @ w_prime @ kk_x_prime)[0, 0]


# The following function evaluates the function H^2 conditioned on arg (see Sections 5.2)
def hscic(arg, k_x, k_y, z, w, sigma_z):
    kk_z = kernel_matrix(z, np.reshape(arg, [1, -1]), sigma_z)
    first = kk_z.T @ w @ np.multiply(k_x, k_y) @ w @ kk_z
    second = kk_z.T @ w @ np.multiply(k_x @ w @ kk_z, k_y @ w @ kk_z)
    third = (kk_z.T @ w @ k_x @ w @ kk_z) * (kk_z.T @ w @ k_y @ w @ kk_z)
    return (first - 2 * second + third)[0, 0]


# We use the following function in simulating the data
def f_a(a, z):
    return np.exp(-0.5 * np.power(z, 2)) * np.sin(a * z)


# Set the hyperparameters.
np.random.seed(22)
n = 500
sigma_X = 0.1
sigma_Y = 0.1
sigma_Z_mcmd = 0.1
sigma_Z_hscic = 0.1
lamb = 0.01

# In the following, we simulate our data for MCMD experiments, as well as their respective kernel matrices.
Z = np.random.normal(size=n)
Z_prime = np.random.normal(size=n)
K_Z_mcmd = kernel_matrix(Z, sigma=sigma_Z_mcmd)
K_Z_prime_mcmd = kernel_matrix(Z_prime, sigma=sigma_Z_mcmd)
W_mcmd = np.linalg.inv(K_Z_mcmd + lamb * np.identity(n))
W_prime_mcmd = np.linalg.inv(K_Z_prime_mcmd + lamb * np.identity(n))
X = f_a(2, Z) + 0.3 * np.random.normal(size=n)
K_X = kernel_matrix(X, sigma=sigma_X)
X_prime_same = f_a(2, Z_prime) + 0.3 * np.random.normal(size=n)
K_X_prime_same = kernel_matrix(X_prime_same, sigma=sigma_X)
K_X_X_prime_same = kernel_matrix(X, X_prime_same, sigma_X)
X_prime_diff = Z_prime + 0.3 * np.random.normal(size=n)
K_X_prime_diff = kernel_matrix(X_prime_diff, sigma=sigma_X)
K_X_X_prime_diff = kernel_matrix(X, X_prime_diff, sigma_X)

# We now simulate our data for HSCIC experiments, as well as their respective kernel matrices.
# These are simulated from the multiplicative noise model, to go in the main body of the text
K_Z_hscic = kernel_matrix(Z, sigma=sigma_Z_hscic)
W_hscic = np.linalg.inv(K_Z_hscic + lamb * np.identity(n))
X_hscic = f_a(2, Z) * np.random.normal(size=n)
K_X_hscic = kernel_matrix(X_hscic, sigma=sigma_X)
Y_ind = f_a(2, Z) * np.random.normal(size=n)
K_Y_ind = kernel_matrix(Y_ind, sigma=sigma_Y)
Y_dep = f_a(2, Z) * np.random.normal(size=n) + 0.2 * X_hscic
K_Y_dep = kernel_matrix(Y_dep, sigma=sigma_Y)
Y_dep_prime = f_a(2, Z) * np.random.normal(size=n) + 0.4 * X_hscic
K_Y_dep_prime = kernel_matrix(Y_dep_prime, sigma=sigma_Y)

# We also simulate from the additive noise model.
X_add = f_a(2, Z) + 0.3 * np.random.normal(size=n)
K_X_add = kernel_matrix(X_add, sigma=sigma_X)
Y_noise = 0.3 * np.random.normal(size=n)
K_Y_noise = kernel_matrix(Y_noise, sigma=sigma_Y)
Y_dep_add = f_a(2, Z) + 0.3 * np.random.normal(size=n) + 0.2 * X_add
K_Y_dep_add = kernel_matrix(Y_dep_add, sigma=sigma_Y)
Y_dep_add_prime = f_a(2, Z) + 0.3 * np.random.normal(size=n) + 0.4 * X_add
K_Y_dep_add_prime = kernel_matrix(Y_dep_add_prime, sigma=sigma_Y)

# Now we compute the MCMD values, conditional witness function values and HSCIC values
z_arguments_mcmd = np.arange(-3, 3, 0.1)
mcmd_same = np.asarray([mcmd(p, K_X, K_X_prime_same, K_X_X_prime_same, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_Z_mcmd)
                        for p in z_arguments_mcmd])
mcmd_diff = np.asarray([mcmd(p, K_X, K_X_prime_diff, K_X_X_prime_diff, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_Z_mcmd)
                        for p in z_arguments_mcmd])
x_arguments = np.arange(-3, 3, 0.1)
zz = np.linspace(-3, 3, 60)
xx = np.linspace(-3, 3, 60)
ZZ, XX = np.meshgrid(zz, xx)
wit_same = np.asarray([[witness(p, q, X, X_prime_same, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_X, sigma_Z_mcmd)
                        for p in z_arguments_mcmd] for q in x_arguments])
wit_diff = np.asarray([[witness(p, q, X, X_prime_diff, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_X, sigma_Z_mcmd)
                        for p in z_arguments_mcmd] for q in x_arguments])
maxi = max(np.max(wit_same), np.max(wit_diff))
mini = min(np.min(wit_same), np.min(wit_diff))
z_arguments_hscic = np.arange(-2, 2, 0.1)
hscic_ind = np.asarray([hscic(p, K_X_hscic, K_Y_ind, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_dep = np.asarray([hscic(p, K_X_hscic, K_Y_dep, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_dep_prime = np.asarray([hscic(p, K_X_hscic, K_Y_dep_prime, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_noise = np.asarray([hscic(p, K_X_add, K_Y_noise, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_dep_add = np.asarray([hscic(p, K_X_add, K_Y_dep_add, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_dep_add_prime = np.asarray([hscic(p, K_X_add, K_Y_dep_add_prime, Z, W_hscic, sigma_Z_hscic)
                                  for p in z_arguments_hscic])


plot_indices = np.sort(np.random.choice(500, 200))
# Plot of MCMD(X,X'_same|Z) and MCMD(X,X'_diff|Z), as well as heat maps showing the conditional witness functions.
# This plot is shown as Figure 2 in the main body of the paper.
fig_2, axes_2 = plt.subplots(1, 4)
axes_2[0].scatter(Z[plot_indices], X[plot_indices], label="X")
axes_2[0].scatter(Z[plot_indices], X_prime_same[plot_indices], label="X'_same", marker="^")
axes_2[0].scatter(Z[plot_indices], X_prime_diff[plot_indices], label="X'_diff", marker="x")
axes_2[0].legend(fontsize=13)
axes_2[0].set_title('Simulated Data', fontsize=20)
axes_2[0].set_xlabel("z", fontsize=20)
axes_2[0].set_ylabel("x", fontsize=20)
axes_2[1].plot(z_arguments_mcmd, mcmd_same, label="MCMD(X,X'_same|Z)", linewidth=5, color="orange")
axes_2[1].plot(z_arguments_mcmd, mcmd_diff, label="MCMD(X,X'_diff|Z)", linestyle="dashed", linewidth=5, color="green")
axes_2[1].legend(fontsize=13)
axes_2[1].set_title('MCMD values', fontsize=20)
axes_2[1].set_xlabel("z", fontsize=20)
axes_2[1].set_ylabel("MCMD", fontsize=20)
im_same = axes_2[2].imshow(wit_same, cmap='viridis', interpolation='nearest', vmin=mini, vmax=maxi,
                           extent=[z_arguments_mcmd[0], z_arguments_mcmd[-1], x_arguments[0], x_arguments[-1]])
axes_2[2].set_title("Witness between X and X'_same", fontsize=20)
colour_bar_same = axes_2[2].figure.colorbar(im_same, ax=axes_2[2])
axes_2[2].set_xlabel("z", fontsize=20)
axes_2[2].set_ylabel("x", fontsize=20)
im_diff = axes_2[3].imshow(wit_diff, cmap='viridis', interpolation='nearest', vmin=mini, vmax=maxi,
                           extent=[z_arguments_mcmd[0], z_arguments_mcmd[-1], x_arguments[0], x_arguments[-1]])
axes_2[3].set_title("Witness between X and X'_diff", fontsize=20)
colour_bar = axes_2[3].figure.colorbar(im_diff, ax=axes_2[3])
axes_2[3].set_xlabel("z", fontsize=20)
axes_2[3].set_ylabel("x", fontsize=20)

# Plot of HSCIC(X,Y_ind|Z), HSCIC(X,Y_dep|Z) and HSCIC(X,Y'_dep|Z), both for additive and multiplicative noise.
# This is shown as Figure 3 in the main body of the paper.
fig_3, axes_3 = plt.subplots(1, 4)
axes_3[0].scatter(Z[plot_indices], X_add[plot_indices], label="X")
axes_3[0].scatter(Z[plot_indices], Y_noise[plot_indices], label="Y_noise", marker="*")
axes_3[0].scatter(Z[plot_indices], Y_dep_add[plot_indices], label="Y_dep_add", marker="x")
axes_3[0].scatter(Z[plot_indices], Y_dep_add_prime[plot_indices], label="Y'_dep_add", marker="D")
axes_3[0].legend(fontsize=13)
axes_3[0].set_title('Simulated Data - Additive Noise', fontsize=20)
axes_3[0].set_xlabel("z", fontsize=20)
axes_3[0].set_ylabel("x, y", fontsize=20)
axes_3[1].plot(z_arguments_hscic, hscic_noise, label="HSCIC(X,Y_noise|Z)", linewidth=5, color="orange")
axes_3[1].plot(z_arguments_hscic, hscic_dep_add, label="HSCIC(X,Y_dep_add|Z)", linestyle="dashed", linewidth=5,
               color="green")
axes_3[1].plot(z_arguments_hscic, hscic_dep_add_prime, label="HSCIC(X,Y'_dep_add|Z)", linestyle="dotted", linewidth=5,
               color="red")
axes_3[1].legend(fontsize=13)
axes_3[1].set_title('HSCIC values', fontsize=20)
axes_3[1].set_xlabel("z", fontsize=20)
axes_3[1].set_ylabel("HSCIC", fontsize=20)
axes_3[2].scatter(Z[plot_indices], X[plot_indices], label="X")
axes_3[2].scatter(Z[plot_indices], Y_ind[plot_indices], label="Y_ind", marker="^")
axes_3[2].scatter(Z[plot_indices], Y_dep[plot_indices], label="Y_dep", marker="x")
axes_3[2].scatter(Z[plot_indices], Y_dep_prime[plot_indices], label="Y'_dep", marker="D")
axes_3[2].legend(fontsize=13)
axes_3[2].set_title('Simulated Data - Multiplicative Noise', fontsize=20)
axes_3[2].set_xlabel("z", fontsize=20)
axes_3[2].set_ylabel("x, y", fontsize=20)
axes_3[3].plot(z_arguments_hscic, hscic_ind, label="HSCIC(X,Y_ind|Z)", linewidth=5, color="orange")
axes_3[3].plot(z_arguments_hscic, hscic_dep, label="HSCIC(X,Y_dep|Z)", linestyle="dashed", linewidth=5, color="green")
axes_3[3].plot(z_arguments_hscic, hscic_dep_prime, label="HSCIC(X,Y'_dep|Z)", linestyle="dotted", linewidth=5,
               color="red")
axes_3[3].legend(fontsize=13)
axes_3[3].set_title('HSCIC values', fontsize=20)
axes_3[3].set_xlabel("z", fontsize=20)
axes_3[3].set_ylabel("HSCIC", fontsize=20)

plt.show()
