"""Device-resident port of reference/pilot_cpu_interface/matsuoka.py.

Fixed-frequency Liu-method migration, not a full PPOC reproduction.
"""
import math
import torch

__all__ = ['Matsuoka']


class Matsuoka:
    def __init__(self, n, device='cuda', dtype=torch.float64, c=.75, kf=1.):
        self.c, self.kf = c, kf
        self.w = torch.zeros(8, 8, device=device, dtype=dtype)
        for i in range(3):
            for side in range(2):
                self.w[2*(i+1)+side, 2*i+1-side] = 8.8669
                self.w[2*i+side, 2*(i+1)+side] = .7844
        self.initial = torch.zeros(n, 4, 2, 2, device=device, dtype=dtype)
        self.initial[:, :, 0, 0] = torch.tensor([.1, -.05, .02, -.01], device=device, dtype=dtype)
        self.state = self.initial.clone()

    def reset(self, mask):
        self.state.copy_(torch.where(mask[:, None, None, None], self.initial, self.state))

    def rhs(self, state, alpha):
        x, y = state[..., 0], state[..., 1]
        z = x.clamp_min(0)
        ue = torch.sigmoid(alpha.clamp(-30, 30))
        u = torch.stack((ue, 1-ue), -1)
        coupling = (y.flatten(1) @ self.w.T).reshape_as(y)
        dx = (-x-4.6062*z.flip(-1)-10.0355*y-coupling+u+self.c)/(.7696*self.kf)
        dy = (z-y)/(1.7728*self.kf)
        return torch.stack((dx, dy), -1)

    def step(self, alpha, dt=.02):
        steps = math.ceil(dt/.0025)
        h = dt/steps
        for _ in range(steps):
            s = self.state
            a = self.rhs(s, alpha)
            b = self.rhs(s+h*a/2, alpha)
            c = self.rhs(s+h*b/2, alpha)
            d = self.rhs(s+h*c, alpha)
            self.state = s+h*(a+2*b+2*c+d)/6
        z = self.state[..., 0].clamp_min(0)
        return 2.0935*(z[..., 0]-z[..., 1])
