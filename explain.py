# from src_dev import *
import torch
from src_dev.LRP import LRP
from src_dev.util import model_io,load_from,write_to
import yaml
from src_dev.model.GraphDataset import GraphDataset
from src_dev.model.InteractionNetwork import InteractionNetwork
from tqdm import tqdm
from torch_geometric.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




if __name__=="__main__":
    # todo: move this to data_io.py
    with open('./data/definitions.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        definitions = yaml.load(file, Loader=yaml.FullLoader)
    
    features = definitions['features']
    spectators = definitions['spectators']
    labels = definitions['labels']

    nfeatures = definitions['nfeatures']
    nspectators = definitions['nspectators']
    nlabels = definitions['nlabels']
    ntracks = definitions['ntracks']


    file_names=["/teams/DSC180A_FA20_A00/b06particlephysics/train/ntuple_merged_0.root"]
    graph_dataset = GraphDataset('./data', features, labels, spectators, n_events=10000, n_events_merge=1000, 
                                file_names=file_names)
    
    # batch=graph_dataset[0]
    # batch_size=1
    # batch_loader=DataLoader(batch,batch_size = batch_size)
    # g=b[1]
    # g.batch=torch.zeros(g.x.shape[0],dtype=torch.int64)

    model=InteractionNetwork().to(device)
    state_dict=torch.load("./data/model/IN_best_dec10.pth",map_location=device)
    model=model_io(model,state_dict,dict())

    t_batch=enumerate(graph_dataset)
    for j,batch in t_batch:
        batch_size=1
        batch_loader=DataLoader(batch,batch_size = batch_size)

        t=tqdm(enumerate(batch_loader),total=len(batch)//batch_size)
        explainer=LRP(model)
        results=[]

        for i,data in t:
            data=data.to(device)
            to_explain={"A":dict(),"inputs":dict(x=data.x,
                                                edge_index=data.edge_index,
                                                batch=data.batch),"y":data.y,"R":dict()}
            
            model.set_dest(to_explain["A"])
            
            results.append(explainer.explain(to_explain,save=False,return_result=True,
            signal=torch.tensor([0,1],dtype=torch.float32).to(device)))
            
        save_to="./data/file_{}_relevance.pt".format(j)
        torch.save(results,save_to)

        # if cnt==0:
        #     break
        # cnt-=1
        
        # break

    # to_explain={"A":dict(),"inputs":dict(x=g.x,edge_index=g.edge_index,batch=g.batch),"R":dict()}
    
    # model=InteractionNetwork().to(device)
    # state_dict=torch.load("./data/model/IN_best_dec10.pth",map_location=torch.device('cpu'))
    # state_dict=torch.load("./data/model/IN_best_dec10.pth",map_location=device)
    # model=model_io(model,state_dict,to_explain["A"])

    # explainer=LRP(model)
    # explainer.explain(to_explain,save=True,save_to="./data/file_0_jet_0_relevance.pt")