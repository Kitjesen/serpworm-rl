"""Liu arXiv:2207.04899v2 Eq.1/3/5 and Table II, fixed-Kf first stage.

Figure 3 wiring: descending cross-side inhibition, ascending same-side
inhibition. Table II weights are assigned 8.8669 descending, .7844 ascending.
This explicit graph transcription is not verified against author software.
"""
import numpy as np


class Matsuoka:
    def __init__(self, n=1, c=.75, kf=1., h=.0025):
        self.c, self.kf, self.h = c, kf, h
        self.w = np.zeros((8, 8))  # target neuron, source neuron
        for i in range(3):
            for side in range(2):
                self.w[2*(i+1)+side, 2*i+(1-side)] = 8.8669
                self.w[2*i+side, 2*(i+1)+side] = .7844
        self.state = np.zeros((n, 4, 2, 2))  # node, e/f, x/y
        self.reset(range(n))

    def reset(self, indices):
        for i in indices:
            self.state[i] = 0
            self.state[i, :, 0, 0] = [.1, -.05, .02, -.01]

    def rhs(self, s, alpha):
        x, y = s[..., 0], s[..., 1]
        z = np.maximum(x, 0)
        ue = 1/(1+np.exp(-np.clip(alpha, -30, 30)))
        u = np.stack([ue, 1-ue], -1)
        coupling = (y.reshape(len(y),8)@self.w.T).reshape(y.shape)
        dx = (-x-4.6062*z[..., ::-1]-10.0355*y-coupling+u+self.c)/(.7696*self.kf)
        dy = (z-y)/(1.7728*self.kf)
        return np.stack([dx, dy], -1)

    def step(self, alpha, dt=.02):
        steps = int(np.ceil(dt/self.h)); h = dt/steps
        for _ in range(steps):
            s = self.state
            a = self.rhs(s, alpha); b = self.rhs(s+h*a/2, alpha)
            c = self.rhs(s+h*b/2, alpha); d = self.rhs(s+h*c, alpha)
            self.state = s+h*(a+2*b+2*c+d)/6
        z = np.maximum(self.state[..., 0], 0)
        return 2.0935*(z[..., 0]-z[..., 1])
