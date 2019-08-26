import numpy as np

class SOM(object):
    def __init__(self, N, K, sigma_max, sigma_min, tau, X, zeta):
        self.t = 0
        self.N = N
        self.K = K
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau
        self.k = np.random.randint(self.K ** 2, size=(self.N ** 2))
        self.X = X
        self.zeta = zeta
        self.hist = {}

    def fit(self, y):
        self.y = y
        sigma_t = max(self.sigma_max - (self.sigma_max - self.sigma_min) * self.t / self.tau, self.sigma_min)

        # 潜在変数Zの推定
        self.k = np.argmin(np.sum((self.X[:, None, :] - self.y[None, :, :]) ** 2, axis=2), axis=1)

        # 参照ベクトルYの推定
        h = np.exp(-1 / (2 * (sigma_t ** 2)) * np.sum((self.zeta[self.k][None, :, :] - self.zeta[:, None, :]) ** 2, axis=2))  # h : K×N
        G = np.diag(np.sum(h, axis=1))
        self.y = np.linalg.inv(G) @ h @ self.X

        self.t += 1
        self.hist.setdefault('y', []).append(self.y)
        self.hist.setdefault('k', []).append(self.k)

    def history(self):
        return self.hist

class SOM2(object):
    def __init__(self, I, L, D, sigma_max, sigma_min, tau, zeta):
        self.t = 0
        self.I = I
        self.L = L
        self.D = D
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau
        self.l = np.random.randint(self.L**2, size=self.I)   # l : I×1
        print(self.l.shape)
        self.zeta = zeta
        self.hist = {}
        self.w_li = []

    def fit(self, y):
        self.sigma_t = max(self.sigma_max - (self.sigma_max - self.sigma_min) * self.t / self.tau, self.sigma_min)

        y_shape = np.shape(y)
        v = np.reshape(y, (y_shape[0], y_shape[1]*y_shape[2]))   # v : I×KD
        print('v.shape:{}'.format(v.shape))

        # 参照ベクトルYの推定
        h = np.exp(-1 / (2 * (self.sigma_t ** 2)) * np.sum((self.zeta[self.l][None,:,:] - self.zeta[:,None,:])**2, axis=2))  # h : L×I
        print('h.shape:{}'.format(h.shape))
        G = np.diag(np.sum(h, axis=1))  # g : I×1
        print('G.shape:{}'.format(G.shape))
        self.w = np.linalg.inv(G) @ h @ v   # w : L×KD
        print('w.shape:{}'.format(self.w.shape))

        # 潜在変数Zの推定
        self.l = np.argmin(np.sum((v[:, None, :] - self.w[None, :, :]) ** 2, axis=2), axis=1) # l : I×1
        print('l.shape:{}'.format(self.l.shape))

        # コピーバック
        self.w_li = self.w[self.l] # w_li : I×D
        print('self.w_li.shape:{}'.format(self.w_li.shape))

        self.t += 1
        self.hist.setdefault('w', []).append(self.w)
        self.hist.setdefault('l', []).append(self.l)

    def history(self):
        return self.hist
