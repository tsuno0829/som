#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
from datasets.datasets import make_hyperbolic_paraboloid_tsom
#from KSE.lib.datasets.artificial.spiral import create_data
#from KSE.lib.datasets.artificial.swiss_roll_2d_mani import create_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

# データの生成
X = make_hyperbolic_paraboloid_tsom(30, 30)
#X = create_data(100, 100)
print(X.shape)

Kx = 20
Ky = 1
K = Kx * Ky
Lx = 20
Ly = 1
L = Lx * Ly
I = X.shape[0]
J = X.shape[1]
D = X.shape[2]
sigma1_max = 1.2
sigma1_min = 0.1
sigma2_max = 1.2
sigma2_min = 0.1
T = 150
tau1 = T - 20
tau2 = T - 20
interval = 100

# モデルの初期化
Y = np.random.rand(K, L, D) * 2.0 - 1.0   # -1.0以上，1.0未満の一様乱数の生成
U = np.random.rand(I, L, D) * 2.0 - 1.0   # U1
V = np.random.rand(K, J, D) * 2.0 - 1.0   # U2

# 潜在空間の生成
mode1_x = np.linspace(-1, 1, Kx)
mode1_y = np.linspace(-1 ,1, Ky)
mode2_x = np.linspace(-1, 1, Lx)
mode2_y = np.linspace(-1, 1, Ly)
mode1_Zeta1, mode1_Zeta2 = np.meshgrid(mode1_x, mode1_y)
mode2_Zeta1, mode2_Zeta2 = np.meshgrid(mode2_x, mode2_y)
Zeta1 = np.c_[mode1_Zeta1.ravel(), mode1_Zeta2.ravel()]   # np.stack + np.reshape = np.c_
Zeta2 = np.c_[mode2_Zeta1.ravel(), mode2_Zeta2.ravel()]

k_star = np.random.randint(K, size=I)
l_star = np.random.randint(L, size=J)
print('k_star.shape:{}'.format(k_star.shape))

k_allepoch = np.zeros((T, I), dtype='int')
l_allepoch = np.zeros((T, J), dtype='int')
Y_allepoch = np.zeros((T, K, L, D))

for t in range(T):
    print('{}回目'.format(t+1))

    sigma1_t = max(sigma1_max - (sigma1_max - sigma1_min) * t / tau1, sigma1_min)
    sigma2_t = max(sigma2_max - (sigma2_max - sigma2_min) * t / tau2, sigma2_min)

    R1 = np.exp(-1 / (2 * (sigma1_t ** 2)) * np.sum((Zeta1[:, None, :] - Zeta1[k_star][None, :, :]) ** 2, axis=2))   # K×I R1=H1
    R2 = np.exp(-1 / (2 * (sigma2_t ** 2)) * np.sum((Zeta2[:, None, :] - Zeta2[l_star][None, :, :]) ** 2, axis=2))   # L×J R2=H2
    #print('R1:{}'.format(R1))
    #print('R1.shape:{}'.format(R1.shape))
    G1 = np.diag(np.sum(R1, axis=1))
    G2 = np.diag(np.sum(R2, axis=1))
    R1_tilde = np.linalg.inv(G1) @ R1
    R2_tilde = np.linalg.inv(G2) @ R2
    #print('R1_tilde:{}'.format(R1_tilde))
    print('R1_tilde.shape:{}'.format(R1_tilde.shape))
    #print('R2_tilde:{}'.format(R2_tilde))
    print('R2_tilde.shape:{}'.format(R2_tilde.shape))

    #for i in range(I):
    #    for l in range(L):
    #        for d in range(D):
    #            s = np.sum(R2_tilde[l, :] @ X[i, :, d])
    #            U[i][l][d] = s
    U = np.einsum('ijd, lj -> ild', X, R2_tilde)

    #for k in range(K):
    #    for j in range(J):
    #        for d in range(D):
    #            s = np.sum(R1_tilde[k, :] @ X[:, j, d])
    #            V[k][j][d] = s
    V = np.einsum('ijd, ki -> kjd', X, R1_tilde)

    #for k in range(K):
    #    for l in range(L):
    #        for d in range(D):
    #            s = np.sum(U[:, l, d] @ R1_tilde[k, :])
    #            Y[k][l][d] = s
    Y = np.einsum('ild, ki -> kld', U, R1_tilde)

    print('U.shape:{}'.format(U.shape))
    print('V.shape:{}'.format(V.shape))
    k_star = np.argmin(np.sum(np.sum((U[:, None, :, :] - Y[None, :, :, :]) ** 2, axis=3), axis=2), axis=1)   # (I, K, L, D)
    print('k_star.shape:{}'.format(k_star.shape))
    l_star = np.argmin(np.sum(np.sum((V[:, :, None, :] - Y[:, None, :, :]) ** 2, axis=3), axis=0), axis=1)   # (K, J, L, D)
    print('l_star.shape:{}'.format(l_star.shape))

    k_allepoch[t, :] = k_star
    l_allepoch[t, :] = l_star
    Y_allepoch[t, :, :, :] = Y

def update(t, T, fig, ax1, ax2, ax3, X, Zeta1, Zeta2, k_star, l_star, Y_allepoch):
    ax1.cla()
    ax2.cla()
    ax3.cla()

    ax1.scatter(Zeta1[k_star[t]][:, 0], Zeta1[k_star[t]][:, 1])
    ax2.scatter(Zeta2[l_star[t]][:, 0], Zeta2[l_star[t]][:, 1])
    ax3.plot_wireframe(Y_allepoch[t, :, :, 0], Y_allepoch[t, :, :, 1], Y_allepoch[t, :, :, 2])
    ax3.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], marker='+')

    fig.suptitle('[{}/{}epoch]   (Left) Latent Space   (Right) Observation Space'.format(t + 1, T), fontsize=16)


if __name__ == '__main__':
    gridsize = (4, 7)
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid(gridsize, (0, 2), colspan=5, rowspan=4, projection='3d')
    fargs = [T, fig, ax1, ax2, ax3, X, Zeta1, Zeta2, k_allepoch, l_allepoch, Y_allepoch]
    ani = anim.FuncAnimation(fig, update, fargs=fargs, interval=interval, frames=T)
    ani.save("tsom_kura.gif", writer='pillow')
