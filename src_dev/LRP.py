import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
import numpy as np
import json
from .util import copy_tensor,model_io


class LRP:
    EPSILON=1e-9

    def __init__(self,model:model_io):
        self.model=model

    def register_model(model:model_io):
        self.model=model

    """
    LRP rules
    """
    @staticmethod
    def eps_rule(layer,input,R):
        a=copy_tensor(input)
        a.retain_grad()
        z=layer.forward(a)
        s=R/(z+LRP.EPSILON*torch.sign(z))

        (z*s.data).sum().backward()

        c=a.grad
        return a*c

    @staticmethod
    def z_rule(layer,input,R):
        w=copy_tensor(layer.weight.data)
        b=copy_tensor(layer.bias.data)

        def f(x):
            x.retain_grad()
            
            n=x*w
            d=n+b*torch.sign(n)*torch.sign(b)
            
            return n/d


        frac=f(input)
        return frac*R

    
    """
    explanation functions
    """
    def explain_single_layer(self,to_explain,index=None,name=None):
        # todo: deal with special case when previous layer has not been explained

        # preparing variables required for computing LRP
        layer=self.model.get_layer(index=index,name=name)
        rule=self.model.get_rule(index=index,layer_name=name)
        if rule=="z":
            rule=LRP.z_rule
        elif rule=="eps":
            rule=LRP.eps_rule
        else:             # default to use epsilon rule if provided rule name not supported
            rule=LRP.eps_rule

        if name is None:
            name=self.model.index2name(index)
        if index is None:
            index=self.model.name2index(name)

        input=to_explain['A'][name]
        
        R=to_explain["R"][index+1] 
        if name in self.model.special_layers:
            n_tracks=to_explain["inputs"]["x"].shape[0]
            row,col=to_explain["inputs"]["edge_index"]

            if "node_mlp_2.3" in name:
                R=R.repeat(n_tracks,1)/n_tracks
            elif "node_mlp_1.3" in name:
                r_x,r_=R[:,:48],R[:,48:]
                R=r_[col]/(n_tracks-1)
                to_explain["R"]["r_x"]=r_x
            elif "edge_mlp.3" in name:
                r_x_row,r_=R[:,:48],R[:,48:]
                R=r_
                to_explain["R"]["r_x_row"]=r_x_row
            elif "bn" in name:
                r_src,r_dest=R[:,:48],R[:,48:]
                to_explain["R"]['r_src']=r_src
                to_explain["R"]['r_dest']=r_dest

                # aggregate
                r_x_src=scatter_mean(r_src,row,dim=0,dim_size=n_tracks)
                r_x_dest=scatter_mean(r_dest,col,dim=0,dim_size=n_tracks)

                r_x=to_explain['R']['r_x']
                r_x_row=to_explain['R']['r_x_row']

                R=(r_x_src+r_x_dest+r_x+scatter_mean(r_x_row,row,dim=0,dim_size=n_tracks)+1e-10)
            else:
                pass


        # backward pass with specified LRP rule
        R=rule(layer,input,R)

        # store result
        to_explain["R"][index]=R


    def explain(self,
                to_explain:dict,
                save:bool=True,
                save_to:str="./relevance.pt",
                # input:dict,
                signal=torch.tensor([0,1],dtype=torch.float32),
                return_relevance:bool=False):
        inputs=to_explain["inputs"]


        self.model.model.eval()
        u=self.model.model.forward(**inputs)

        start_index=self.model.n_layers
        to_explain['R'][start_index]=copy_tensor(u*signal)

        for index in range(start_index-1,0-1,-1):
            self.explain_single_layer(to_explain,index)

        R_node=to_explain["R"][0]
        R_edge=to_explain["R"]['r_src']+to_explain["R"]['r_dest']+(to_explain["R"]['r_x_row'])
        if save:
            torch.save(dict(node=R_node,edge=R_edge),save_to)
        
        if return_relevance:
            return R_node


