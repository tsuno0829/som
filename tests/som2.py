#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
from models.SOM2 import SOM
from models.SOM2 import SOM2
from datasets.datasets import make_datasets4som2

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as anim
from matplotlib.cm import ScalarMappable

def update(t, fig, ax1, ax2, ax3, X, zeta, zeta_parent, k, y, l):
    plt.cla()
    ax2.cla()
    for i in range(len(X)):
        ax2.scatter(X[i,:,0], X[i,:,1])

    for i, ax in enumerate(ax1):
        ax.cla()
        ax.scatter(zeta[k[i,t]][:,0],zeta[k[i,t]][:,1])
        ax.set_title('(Latent) child_SOM{}'.format(i))
        Y = np.reshape(y[i,t], (K,K,2))
        ax2.scatter(Y[:,:,0], Y[:,:,1])
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.set_title('(Observation space) child_SOM')

    ax3.scatter(zeta_parent[l[t]][:,0], zeta_parent[l[t]][:,1])
    ax3.set_title('(Latent space) parent_SOM')
    fig.suptitle('{}epoch'.format(t+1), fontsize=16)

if __name__ == '__main__':
    np.random.seed(1)
    N = 20           # データ数(N**2)
    K = 15           # 潜在空間のノード数(K**2)
    D = 2            # データ次元
    I = 9            # 親SOMのデータの個数
    L = 15           # 親SOMのユニット数
    T = 30           # 学習回数
    tau = 25         # 時定数(T>tau)
    sigma_max = 2.0  # 近傍半径の最大値
    sigma_min = 0.1  # 近傍半径の最小値
    interval = 300   # アニメーションの描画間隔

    # データ集合の作成
    datasets_path = '../datasets/datasets4som2/*'
    X = make_datasets4som2(datasets_path)
    print(X.shape)

    # 子SOMのノード集合の作成
    zeta = np.dstack(np.meshgrid(np.linspace(-1,1,K),
                                 np.linspace(-1,1,K),
                                 indexing='ij'))
    zeta = np.reshape(zeta, (K**2, D))
    print('zeta.shape:{}'.format(zeta.shape))

    # 親SOMのノード集合の作成
    zeta_parent = np.dstack(np.meshgrid(np.linspace(-1,1,L),
                                        np.linspace(-1,1,L),
                                        indexing='ij'))
    zeta_parent = np.reshape(zeta_parent, (L**2, D))
    print('zeta_parent:{}'.format(zeta_parent.shape))

    # 子SOMの作成
    child_som = [0]*len(X)
    for i in range(len(X)):
        child_som[i] = SOM(N, K, sigma_max, sigma_min, tau, X[i], zeta)
    print(child_som)

    # 親SOMの作成
    parent_som = SOM2(I, L, D, sigma_max, sigma_min, tau, zeta_parent)

    # 学習
    y = np.zeros((len(X), K**2, D))
    for t in range(T):
        print('{}回目'.format(t))
        reference_vector = []   # reference_vector : 9x225x2

        # 子SOMの学習
        for j, som in enumerate(child_som):
            som.fit(y)
            reference_vector.append(som.y)

        # 親SOMの学習
        parent_som.fit(reference_vector)

        # copy back
        print('parent_som.w.shape:{}'.format(parent_som.w_li.shape))
        y = np.reshape(parent_som.w_li, (I, K**2, D))
        print('learned_y.shape:{}'.format(y.shape))

    history_child_k = np.array([som.history()['k'] for som in child_som])
    history_child_y = np.array([som.history()['y'] for som in child_som])
    history_parent_l = np.array(parent_som.history()['l'])
    print(history_child_k.shape, history_child_y.shape, history_parent_l.shape)

    fig = plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    ax1 = []
    for i in range(1,I+1):
        ax1.append(fig.add_subplot(4,3,i))
    ax2 = fig.add_subplot(4,3,10)
    ax3 = fig.add_subplot(4,3,11)
    fargs = [fig, ax1, ax2, ax3, X, zeta, zeta_parent, history_child_k, history_child_y, history_parent_l]
    ani = anim.FuncAnimation(fig, update, fargs=fargs, interval=interval, frames=T)
    ani.save("som2.gif", writer='imagemagick')
    #plt.tight_layout()