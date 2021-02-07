import torch
import torch.nn as nn
from torch.nn import Sequential as Seq,Linear,ReLU,BatchNorm1d
from torch_scatter import scatter_mean
from torch_geometric.utils import to_networkx

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from InteractionNetwork import InteractionNetwork

from util import copy_layer,copy_tensor,LRP
from plot import plot_node_feat_heatmap,plot_network_3D

import yaml
from GraphDataset import GraphDataset

def load_data(def_fn="definitions.yml",
              fn=["/teams/DSC180A_FA20_A00/b06particlephysics/train/ntuple_merged_10.root"]):
    # todo: update to use our own data loader from last quarter
    with open(def_fn) as infile:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        definitions = yaml.load(infile, Loader=yaml.FullLoader)
        
    features = definitions['features']
    spectators = definitions['spectators']
    labels = definitions['labels']

    nfeatures = definitions['nfeatures']
    nspectators = definitions['nspectators']
    nlabels = definitions['nlabels']
    ntracks = definitions['ntracks']

    file_names=fn
    graph_dataset = GraphDataset('data', features, labels, spectators, n_events=10000, n_events_merge=1000, 
                             file_names=file_names)

    return graph_dataset

def load_model(fn):
    model=InteractionNetwork()
    model.load_state_dict(torch.load(fn))

    return model

def get_layers(model)->dict:
    """
    loads the model and creates a collection of the layer slices copy
    @param model: the interaction network model to be dissected 
    """
    layers={}
    cnt=0

    for i in (model.interactionnetwork.children()):
        layer=[]
        for n,j in i.named_modules():
            # todo: update the slicing rule to be imported from file
            if n and n[-1].isnumeric() and ("." in n):
                if n[-1] in ["0","1","2"]:
                    layer.append(copy_layer(j))  # part of the rectified layer to be merged into one
                else:
                    # merge the normalization layers and activation layers with closest linear layer
                    layers[f"L{cnt}"]=Seq(*layer)
                    layer=[]
                    cnt+=1
                    
                    # create a copy of the current linear layer
                    layers[f"L{cnt}"]=copy_layer(j)
                    cnt+=1
    
    return layers

def get_input(g):
    """
    given a graph g, prepare the input matrix for the IN model layers, along
    with required meta information

    @param g: a data graph
    """
    row,col=g.edge_index
    n_tracks=g.x.shape[0]

    x=nn.BatchNorm1d(48)(g.x)
    x=copy_tensor(torch.cat([x[row],x[col]],1))

    return x,g.edge_index

def get_activations(layers,x,edge_index):
    """
    @param layers: the entire collection of layers
    @param x: preprocessed input data
    """
    # initialize
    activations={}
    activations["a-1"]=x
    activations["a0"]=layers["L0"].forward(x)

    # create masking matrices for normalization by src/dest nodes of the edges
    row,col=edge_index
    M_col=torch.zeros(col.shape[0],n_tracks,dtype=torch.float32)
    M_row=torch.zeros(row.shape[0],n_tracks,dtype=torch.float32)
    for i,j in enumerate(col):
        M_col[i,j]=1
    for i,j in enumerate(row):
        M_row[i,j]=1
    M_col=M_col.T
    M_row=M_row.T


    # todo: change the activation and layer extraction to forward hooks
    # propogate through the layer slices and extract activations
    for i in range(1,len(layers.keys())):
        if i==2:
            #a1->a1'
            a=copy_tensor(torch.cat([g.x[row],activations[f"a{i-1}"]],1))           # [x[row],a1]
        elif i==4:
            #a3->a3'
            a=copy_tensor(torch.cat([g.x,M_col@activations[f"a{i-1}"]/n_tracks],1)) # g.x,scatter_mean(a3,col,dim=0)
        elif i==6:
            #a5->a5'
            layers[f"L{i}"].eval()
            a=copy_tensor(torch.ones(1,n_tracks)@activations[f"a{i-1}"]/n_tracks)   # mean of all tracks
        else:
            a=copy_tensor(activations[f"a{i-1}"])
        
        activations[f"a{i}"]=layers[f"L{i}"].forward(a)

    activations["output"]=copy_tensor(activations["a{i}"])

    return activations

def get_relevances(layers,activations):
    R={}
    n_layers=len(activations.keys())

    # mask the output to only look at the contribution to the signal class
    R[f"R{n_layers+1}"]=copy_tensor(activations["output"]@torch.tensor([[0,0],[0,1]],dtype=torch.float32))


    # todo: update the special case rules to be imported from file
    # compute relevance scores
    for i in range(len(layers.keys())-1,-1,-1):
        a=activations[f"a{i-1}"]
        r=R[f"R{i+1}"]
        l=layers[f"L{i}"]
  
    if i==2:
        # a1->a1'
        a=copy_tensor(torch.cat([g.x[row],a],1))
        r=LRP(a,l,r)
        
        # r_x[row],r2'
        r_src,r=r[:,:48],r[:,48:] 
        R[f"R{i}_src"]=r_src

    elif i==4:
        # a3->a3'
        a=copy_tensor(torch.cat([g.x,M_col@a/n_tracks],1))
        r=LRP(a,l,r)
        
        r_x,r=r[:,:48],r[:,48:]
        R[f"R{i}_x"]=r_x
        r=r[col]/n_tracks
    elif i==6:
        # a5->a5'
        a=copy_tensor((torch.ones(1,n_tracks)@a/n_tracks))
        r=LRP(a,l,r)
        
    else:
        a=copy_tensor(a)
        r=LRP(a,l,r)
        
    R[f"R{i}"]=r


def pipeline(graph,
            features,
            model_fn,node_feat_fn,nw_rel_fn):
    """
    a pipeline connecting all the steps for making relevant plots
    """
    # prepare dataset, use only a subset of it for visualization
    graph.batch=torch.tensor(np.zeros(graph.x.shape[0]).astype("int64"))
    x,edge_index=get_input(graph)

    # load model
    model=load_model(model_fn)

    # create layer slices
    layers=get_layers(model)

    # obtain activation of each layer and compute relevance scores
    activations=get_activations(layers,x,edge_index)
    R=get_relevances(layers,activations)

    # merge the relevance scores from multiple paths
    r=(((M_row@R["R0"][:,:len(features)]))+(M_row@R["R2_src"])+R["R4_x"])

    # make feature relevance heatmap at each node
    plot_node_feat_heatmap(r,features,node_feat_fn)

    
    # process relevance scores for making 3D jet plot
    # create masking matrices for normalization by src/dest nodes of the edges
    M_row=torch.zeros(row.shape[0],n_tracks,dtype=torch.float32)
    for i,j in enumerate(row):
        M_row[i,j]=1
    M_row=M_row.T

    node_size=torch.norm(M_row@R["R0"][:,:48],dim=1)
    edge_alpha=torch.norm(R["R0"],dim=1)
    alpha=(edge_alpha-edge_alpha.min())/edge_alpha.max() # normalize so edge alpha varies between 0-1
    alpha=alpha.detach().numpy()
    
    # add the computed attributes to the networkx graph
    graph.edge_alpha=edge_alpha
    graph.node_size=node_size

    eta_idx=features.index("track_etarel")
    phi_idx=features.index("track_phirel")
    pt_idx =features.index("track_pt")
    pos=np.array(list(zip(graph.x[:,eta_idx].detach().numpy(),
                          graph.x[:,phi_idx].detach().numpy(),
                          graph.x[:,pt_idx].detach().numpy())))
    graph.pos=pos
    G = to_networkx(graph, edge_attrs=['edge_alpha'],node_attrs=["pos","node_size"])

    # plot edge relevance 3D
    network_plot_3D(G,45,graph.y.detach(),fn=nw_rel_fn)
    

if __name__=="__main__":
    # get feature names for plotting
    with open("definitions.yml") as file:
        definitions = yaml.load(file, Loader=yaml.FullLoader)
        
    features = definitions['features']

    dataset=load_data()
    b=dataset[0]
    g=b[0]
    
    pipeline(g,features,
            "../data/model/IN_best_dec10.pth",
            "IN_best_dec10_node_feat_relevance.png",
            "IN_best_dec10_nw_rel_3d.png"
            )
