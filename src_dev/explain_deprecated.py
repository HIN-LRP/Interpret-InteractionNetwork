from src_dev.LRP import LRP
from src_dev.util import model_io,load_from,write_to
import yaml
from src_dev.model.GraphDataset import GraphDataset
from src_dev.model.InteractionNetwork import InteractionNetwork
import torch
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


    file_names=["/teams/DSC180A_FA20_A00/b06particlephysics/train/ntuple_merged_10.root"]
    graph_dataset = GraphDataset('./data', features, labels, spectators, n_events=10000, n_events_merge=1000, 
                                file_names=file_names)
    
    b=graph_dataset[0]
    g=b[1]
    g.batch=torch.tensor(np.zeros(g.x.shape[0]).astype("int64"))

    to_explain={"A":dict(),"inputs":dict(x=g.x,edge_index=g.edge_index,batch=g.batch)}
    
    model=InteractionNetwork()
    state_dict=torch.load("./data/model/IN_best_dec10.pth",map_location=torch.device('cpu'))
    model=model_io(state_dict,to_explain)

    explainer=LRP(model)
    explainer.explain(to_explain,save=True,save_to="./data/file_0_jet_0_relevance.pt")