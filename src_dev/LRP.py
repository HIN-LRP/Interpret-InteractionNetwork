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
        s=R/(z+EPSILON*torch.sign(z))

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
        
        return f

        frac=f(a)
        return frac*R

    
    """
    explanation functions
    """
    def explain_single_layer(self,to_explain,index=None,name=None):
        # todo: deal with special case when previous layer has not been explained

        # preparing variables required for computing LRP
        layer=self.model.get_layer(index=index,name=name)
        rule=self.model.get_rule(index=index,name=name)
        if rule=="z":
            rule=z_rule
        elif rule=="eps":
            rule=eps_rule
        else:             # default to use epsilon rule if provided rule name not supported
            rule=eps_rule


        input=to_explain['A'][name]
        
        R=to_explain["R"][index+1] 
        if name in self.special_layers:
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
                continue
            # to_split=self.special_layer_config[name]["split"]
            # if len(to_split)==1: # do not split
            #     continue
            # else:
            #     for r_n in to_split.keys():
            #         start,stop=to_split[r_n]
            #         if r_n==name:
            #             temp=R[:,start:stop]
            #         else:
            #             to_explain["R"][r_n]=R[:,start:stop]
                
            #     R=temp


            # to_scatter=self.special_layer_config[name]["scatter"]
            # if to_scatter<0:     # do not scatter
            #     continue
            # elif to_scatter==2:  # scatter by edges/tracks
            #     n_tracks=to_explain["inputs"]["x"].shape[0]
            #     R=R.repeat(n_tracks,1)/n_tracks
            # else:                # scatter by row/src or col/dest
            #     R=R[to_explain["inputs"]["edge_index"][to_scatter]]
            #     n_tracks=to_explain["inputs"]["x"].shape[0]
            #     R/=n_tracks

            # to_scatter_mean=self.special_layer_config[name]["scatter_mean"]
            # if len(to_scatter_mean)==0: # do not aggregate
            #     continue
            # else:



        # backward pass with specified LRP rule
        R=rule(layer,input,R)

        # store result
        to_explain["R"][index]=R


    def explain(self,
                to_explain:dict,
                save=True:bool,
                save_to="./relevance.pt":str,
                # input:dict,
                # signal=torch.tensor([0,1],dtype=torch.float32),
                return_relevance=False:bool):
        inputs=to_explain["inputs"]

        self.model.model.eval()
        u=self.model.model.forward(**inputs)

        start_index=len(self.model.n_layers)
        to_explain['R'][start_index]=copy_tensor(u*signal)

        for index in range(start_index-1,1-1,-1):
            self.explain_single_layer(to_explain,index)

        R_node=to_explain["R"][0]
        R_edge=to_explain["R"]['r_src']+to_explain["R"]['r_dest']+(to_explain["R"]['r_x_row'])
        if save:
            torch.save(dict(node=R_node,edge=R_edge),save_to)
        
        if return_relevance:
            return R_node


