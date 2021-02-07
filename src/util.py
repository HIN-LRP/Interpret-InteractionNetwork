import torch
import torch.nn as nn

def copy_layer(layer):
    """
    create a deep copy of provided layer
    """
    layer_cp=eval("nn."+layer.__repr__())
    layer_cp.load_state_dict(layer.state_dict())
    
    return layer_cp


def copy_tensor(tensor,dtype=torch.float32):
    """
    create a deep copy of the provided tensor,
    outputs the copy with specified dtype 
    """
    
    return torch.tensor(tensor.clone().detach().numpy(),
                        requires_grad=True,
                        dtype=dtype)

def LRP(a,l,r,epsilon=1e-9):
    a=torch.tensor(a.clone().detach().numpy(),
                      requires_grad=True,dtype=torch.float32)
    a.retain_grad()
    
    z=l.forward(a)
    s=r/(z+epsilon)
    
    (z*s.data).sum().backward()
    c=a.grad
    
    return a*c

