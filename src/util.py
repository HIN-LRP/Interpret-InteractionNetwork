import torch
import torch.nn as nn

def copy(layer):
    """
    create a deep copy of provided layer
    """
    layer_cp=eval("nn."+layer.__repr__())
    layer_cp.load_state_dict(layer.state_dict())
    
    return layer_cp


def rho(layer,rule="0"):
    """
    create a deep copy of a linear layer
    """
    W=layer.weight.detach()
    b=layer.bias.detach()
    
    return lambda a: a@W.T+b


def lrp(layer,a,R,epsilon=1e-9,relu=False):
    """
    LRP-epsilon, with epsilon set to 1e-9 by default
    """
    a=torch.tensor(a,requires_grad=True)
    a.retain_grad()
    
    z=epsilon+rho(layer)(a)
    s=R/(z+1e-9)
    (z@s.data).sum().backward()
    c=a.grad
    
    if relu:
        R=nn.ReLU()(z.sum())*c
    else:
        R=a*c
    
    return R

