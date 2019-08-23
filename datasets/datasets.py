import glob
import numpy as np
from sklearn import datasets

def make_hyperbolic_paraboloid(N, rng, random_state=True):
    # N            : データ数
    # rng          : 幅，等方となるように設定
    # random_state : 乱数
    data_space_X, data_space_Y = np.meshgrid(np.linspace(-rng,rng,N), np.linspace(-rng,rng,N))
    data_space_Z = data_space_X**2 - data_space_Y**2
    X = np.stack((data_space_X, data_space_Y, data_space_Z),axis=-1)
    X = np.reshape(X,(N**2,3))
    X[:,0] = np.random.random(N**2) * 2 -1
    X[:,1] = np.random.random(N**2) * 2 -1
    X[:,2] = X[:,0]**2 - X[:,1]**2
    return X

def make_swiss_roll(n_samples, noise, random_state):
    return datasets.make_swiss_roll(n_samples, noise, random_state)

def make_datasets4som2(datasets_path):
    # e.g.
    # datasets_path = '../datasets/datasets4som2/*'
    path = sorted(glob.glob(datasets_path))
    X = np.array([np.loadtxt(p) for p in path])
    return X