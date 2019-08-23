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
import matplotlib.animation as anim

def update(t, T, D, I, fig, ax1, ax2, ax3, ax4, X, zeta, zeta_parent, k, y, l, y_gen, l_gen):
    plt.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    for i in range(len(X)):
        ax2.scatter(X[i,:,0], X[i,:,1], c=X[i,:,0])

    for i, ax in enumerate(ax1):
        Y = np.reshape(y[i,t], (K,K,2))
        ax2.scatter(Y[:,:,0], Y[:,:,1])
        ax.cla()
        ax.scatter(zeta[k[i,t]][:,0], zeta[k[i,t]][:,1], c=X[i,:,0])
        ax.set_title('child_SOM{}'.format(i + 1), fontsize=9)
        ax3.scatter(zeta_parent[l[t]][i,0], zeta_parent[l[t]][i,1], s=120)

    if t+1 == T:
        ax3.scatter(zeta_parent[l_gen][:,0], zeta_parent[l_gen][:,1], s=120, marker='*', c='k', label='new sampling')

        XX, YY = np.meshgrid(np.arange(K), np.arange(K))
        #print(XX)
        for i in range(I):
            Y = np.reshape(y[i,t], (K,K,D))
            ax4.scatter(Y[:,:,0], Y[:,:,1],c=XX, cmap='RdYlGn')

        for i in range(len(y_gen)):
            Y_gen = np.reshape(y_gen[i], (K,K,D))
            ax4.scatter(Y_gen[:,:,0], Y_gen[:,:,1], s=60, marker='*', c=XX, cmap='RdYlGn', label='generated data')

    ax3.set_xlim(-1.1,1.1)
    ax3.set_ylim(-1.1,1.1)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax4.set_xlabel("X-axis")
    ax4.set_ylabel("Y-axis")
    ax2.set_title('Data + Reference Vector')
    ax3.set_title('Parent SOM')
    ax4.set_title('Reference Vector( + Generated ☆ )')

    fig.suptitle('[{}epoch]   (Left) Latent Space   (Right) Observation Space'.format(t+1), fontsize=16)

if __name__ == '__main__':
    np.random.seed(1)
    N = 20                  # データ数(N**2)
    K = 25                  # 潜在空間のノード数(K**2)
    D = 2                   # データ次元
    I = 9                   # 親SOMのデータの個数
    L = 10                  # 親SOMのユニット数
    T = 20                  # 学習回数
    tau = 19                # 時定数(T>tau)
    sigma_max_child = 1.0   # 近傍半径の最大値
    sigma_min_child = 0.1   # 近傍半径の最小値
    sigma_max_parent = 1.0
    sigma_min_parent = 0.1
    interval = 800          # アニメーションの描画間隔

    # データ集合の作成
    datasets_path = '../datasets/datasets4som2/*'
    X = make_datasets4som2(datasets_path)
    print('X.shape:{}'.format(X.shape))

    # 子SOMのノード集合の作成
    zeta = np.dstack(np.meshgrid(np.linspace(-1,1,K),
                                 np.linspace(-1,1,K)))
    zeta = np.reshape(zeta, (K**2, D))
    print('zeta.shape:{}'.format(zeta.shape))

    # 親SOMのノード集合の作成
    zeta_parent = np.dstack(np.meshgrid(np.linspace(-1,1,L),
                                        np.linspace(-1,1,L)))
    zeta_parent = np.reshape(zeta_parent, (L**2, D))
    print('zeta_parent:{}'.format(zeta_parent.shape))

    # 子SOMの作成
    child_som = [SOM(N, K, sigma_max_child, sigma_min_child, tau, X[i], zeta) for i in range(I)]

    # 親SOMの作成
    parent_som = SOM2(I, L, D, sigma_max_parent, sigma_min_parent, tau, zeta_parent)

    # 学習
    y = np.zeros((I, K**2, D))
    for t in range(T):
        print('{}回目'.format(t+1))
        reference_vector = []

        # 子SOMの学習
        for i, som in enumerate(child_som):
            som.fit(y[i])
            reference_vector.append(som.y)   # reference_vector : IxK**2xD

        # 親SOMの学習
        parent_som.fit(reference_vector)

        # コピーバック
        print('parent_som.w_li.shape:{}'.format(parent_som.w_li.shape))
        y = np.reshape(parent_som.w_li, (I, K**2, D))
        print('learned_y.shape:{}'.format(y.shape))

    history_child_k = np.array([som.history()['k'] for som in child_som])
    history_child_y = np.array([som.history()['y'] for som in child_som])
    history_parent_l = np.array(parent_som.history()['l'])
    print(history_child_k.shape, history_child_y.shape, history_parent_l.shape)

    # 多様体の潜在変数から新しいデータを生成する
    print('学習後の親SOMのl:{}'.format(sorted(history_parent_l[-1])))
    history_parent_w = np.array(parent_som.history()['w'])
    l_gen = np.array(list(range(L**2)))

    w_gen = history_parent_w[-1][l_gen]
    print('w_gen.shape:{}'.format(w_gen.shape))
    y_gen = np.reshape(w_gen, (len(l_gen), K**2, D))
    print('y_gen.shape:{}'.format(y_gen.shape))

    gridsize = (6, 6)
    fig = plt.figure(figsize=(12,10))
    plt.subplots_adjust(wspace=0.6, hspace=1)
    ax1 = []
    for i in range(0,3):
        for j in range(0,3):
            ax1.append(plt.subplot2grid(gridsize, (i, j)))
    ax2 = plt.subplot2grid(gridsize, (0,3), colspan=3, rowspan=3)
    ax3 = plt.subplot2grid(gridsize, (3,0), colspan=3, rowspan=3)
    ax4 = plt.subplot2grid(gridsize, (3,3), colspan=3, rowspan=3)
    fargs = [T, D, I, fig, ax1, ax2, ax3, ax4, X, zeta, zeta_parent, history_child_k, history_child_y, history_parent_l, y_gen, l_gen]
    ani = anim.FuncAnimation(fig, update, fargs=fargs, interval=interval, frames=T)
    ani.save("som2.gif", writer='imagemagick')