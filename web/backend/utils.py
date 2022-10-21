import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Transform(nn.Module):
    """
    Transform target variable to normalized space.
    Moreover, one can inverse the transformation to return to original space.
    
    Args:
        target (numpy or torch array): original target variable.
    """
    def __init__(self, target):
        super(Transform, self).__init__()
        
        self.max = (1 / (target + 0.065)).max()
        
    def forward(self, x):
        # Divide by max to normalize to [0, 1]
        return 1 / (x + 0.065) / self.max
    
    def inverse(self, x):
        # Divide by max to denormalize.
        # Also, have to subtract a constant
        # to make output consistent with the original target.
        return 0.005 * (200 - 13*x) / x / self.max - 0.060775
    
    
class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)