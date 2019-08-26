import glob
import numpy as np
from sklearn import datasets

def make_hyperbolic_paraboloid(N, rng, random_state=True):
    # Input
    # N            : データ数
    # rng          : 幅，等方となるように設定
    # random_state : 乱数
    # Output
    # X            : 3次元配列に後で変更するかも？
    data_space_X, data_space_Y = np.meshgrid(np.linspace(-rng,rng,N), np.linspace(-rng,rng,N))
    data_space_Z = data_space_X**2 - data_space_Y**2
    X = np.stack((data_space_X, data_space_Y, data_space_Z),axis=-1)
    X = np.reshape(X,(N**2,3))
    X[:,0] = np.random.random(N**2) * 2 -1
    X[:,1] = np.random.random(N**2) * 2 -1
    X[:,2] = X[:,0]**2 - X[:,1]**2
    return X

def make_hyperbolic_paraboloid_tsom(xsamples, ysamples, retz=False):
    z1 = np.linspace(-1, 1, xsamples)
    z2 = np.linspace(-1, 1, ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2)
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = x1 ** 2 - x2 ** 2

    X = np.concatenate((x1[:, :, None], x2[:, :, None], x3[:, :, None]), axis=2)
    truez = np.concatenate((z1_repeated[:, :, None], z2_repeated[:, :, None]), axis=2)

    if retz == True:
        return X, truez
    else:
        return X

def make_swiss_roll(n_samples, noise, random_state):
    return datasets.make_swiss_roll(n_samples, noise, random_state)

def make_datasets4som2(datasets_path):
    # e.g.
    # datasets_path = '../datasets/datasets4som2/*'
    path = sorted(glob.glob(datasets_path))
    X = np.array([np.loadtxt(p) for p in path])
    return X