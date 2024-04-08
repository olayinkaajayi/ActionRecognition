"""Hyperboloid manifold."""

import torch
# Might remove all the .clamps
# clamps must stay because result were nan without them
from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature.
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[..., 0:1] + sqrtK)

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        c = c.to(x.device) ##Added
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2 #formerly dim=1
        mask = torch.ones_like(x)
        mask[..., 0] = 0
        vals = torch.zeros_like(x)
        bin_op = K + y_sqnorm
        vals[..., 0:1] = torch.sqrt(torch.clamp(bin_op, min=self.eps[x.dtype]))
        bin_op = vals + mask * x #Added
        return bin_op # vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(-1) - 1 #changed from x.size(1)
        bin_op = x.narrow(-1, 1, d) * u.narrow(-1, 1, d)
        ux = torch.sum(bin_op, dim=-1, keepdim=True) #formerly dim=1
        mask = torch.ones_like(u)
        mask[..., 0] = 0
        vals = torch.zeros_like(u)
        vals[..., 0:1] = ux / torch.clamp(x[..., 0:1], min=self.eps[x.dtype])
        bin_op = vals + mask * u
        return bin_op # vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[..., 0:1] = narrowed
        bin_op = u - vals
        return bin_op # u - vals

    def expmap(self, u, x, c):
        c = c.to(x.device) ##Added
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        bin_op = self.minkowski_dot(x, y) + K
        xy = torch.clamp(bin_op, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        c = c.to(u.device) #should move c to u's device (DONE)
        K = 1. / c
        sqrtK = K ** 0.5
        b = u.size(0) #batch size <added>
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(b,-1, d) #formerly .view(-1, d)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) #formerly dim=1
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[..., 0:1] = sqrtK * cosh(theta)
        res[..., 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        c = c.to(x.device) ##Added
        K = 1. / c
        sqrtK = K ** 0.5
        b = x.size(0) #batch size <added>
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(b,-1, d) #formerly .view(-1, d)
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True) #formerly dim=1
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        bin_op = x[..., 0:1] / sqrtK
        theta = torch.clamp(bin_op, min=1.0 + self.eps[x.dtype])
        res[..., 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u.matmul( m.transpose(-1, -2) )
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):###########NOT CALLED
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        c = c.to(x.device) ##Added
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=self.min_norm) #formerly dim=1
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[..., 0:1] = - y_norm
        v[..., 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[..., 1:], dim=-1, keepdim=True) / sqrtK #formerly dim=1
        res = u - alpha * v
        return self.proj_tan(res, x, c)
