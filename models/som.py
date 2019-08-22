import numpy as np

class SOM(object):
    def __init__(self, N, K, sigma_max, sigma_min, tau):
        self.t = 0
        self.N = N
        self.K = K
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau
        self.hist = {}
        self.k = []
        
    def fit(self, X, zeta):
        sigma_t = max(self.sigma_max - (self.sigma_max - self.sigma_min)*self.t/self.tau, self.sigma_min)
        
        if self.t == 0:
            self.k = np.random.randint(self.K**2, size=(self.N**2))
        
        # 参照ベクトルYの推定
        h = np.exp( -1/(2*(sigma_t**2)) * np.sum((zeta[self.k][None,:,:]-zeta[:,None,:])**2, axis=2) )   # h : K×N
        g = np.sum(h, axis=1)                                                                            # g : K×1
        y = (1/g[:,None]) * h @ X                                                                        # y : K×D
        
        # 潜在変数Zの推定
        # yをself.yに変更する
        self.k = np.argmin( np.sum((X[:,None,:]-y[None,:,:])**2, axis=2), axis=1 )
        
        self.t += 1
        self.hist.setdefault('y',[]).append(y)
        self.hist.setdefault('k',[]).append(self.k)
        
    def history(self):
        return self.hist