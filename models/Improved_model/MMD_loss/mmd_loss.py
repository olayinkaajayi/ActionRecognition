import torch
import torch.nn as nn
import torch.nn.functional as F

class ModMMD(nn.Module):
    """This class implements a variation of the MMD loss"""
    # Review these hyperparameters
    def __init__(self, axx=0.5, bxx=1.,
                        axy=1., bxy=1.,
                        ayy=0.5, byy=1.):
        super(ModMMD, self).__init__()
        self.axx= axx
        self.bxx= bxx
        self.axy= axy
        self.bxy= bxy
        self.ayy= ayy
        self.byy= byy

    def kernel(self, x, y, a=1.0, b=1.0):
        """This is the Radial Basis Function (RBF) kernel."""

        # b is the radius of the area of influence
        # high values is far, low values is close

        dist = torch.cdist(x,y)**2
        return a*torch.exp(-dist/b)


    def forward(self, x, y):
        """x, y: batch_size x dim"""

        n = x.size(0) #num of samples in x
        m = y.size(0) #num of samples in y

        xx_scale = 1./(n*(n-1)) if n!= 1 else 1
        yy_scale = 1./(m*(m-1)) if m!= 1 else 1
        xy_scale = -2./(n*m)

        return (
                xx_scale*self.kernel(x,x, self.axx, self.bxx).sum() +
                xy_scale*self.kernel(x,y, self.axy, self.bxy).sum() +
                yy_scale*self.kernel(y,y, self.ayy, self.byy).sum()
                )
