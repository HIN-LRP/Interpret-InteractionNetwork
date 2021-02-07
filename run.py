from src import *
import yaml
import sys
from os import system

if __name__=="__main__":
    # get the targets
    targets=sys.argv[1:]

    # get feature names for plotting
    with open("data/definitions.yml") as file:
        definitions = yaml.load(file, Loader=yaml.FullLoader)
        
    features = definitions['features']

    if "test" in targets:
        dataset=load_data("data/definitions.yml",fn=["./test/data/test.root"])
    else:
        dataset=load_data("data/definitions.yml")

    b=dataset[0]
    graph=b[0]

    if len(targets)>0 and not (("all" in targets) or ("test" in targets)):
        # prepare dataset
        graph.batch=torch.tensor(np.zeros(graph.x.shape[0]).astype("int64"))

        # create masking matrices for normalization by src/dest nodes of the edges
        n_tracks=graph.x.shape[0]
        row,col=graph.edge_index
        M_row,M_col=make_mask(row,n_tracks),make_mask(col,n_tracks)

        # load model
        model=load_model("data/model/IN_best_dec10.pth")

        # create layer slices
        layers=get_layers(model)

        # obtain activation of each layer and compute relevance scores
        activations=get_activations(layers,graph)
        R=get_relevances(layers,activations,graph)

        # merge the relevance scores from multiple paths
        r=(((M_row@R["R0"][:,:len(features)]))+(M_row@R["R2_src"])+R["R4_x"])
    elif "test" in targets:
        pipeline(graph,features,
            "data/model/IN_best_dec10.pth",
            "IN_best_dec10_node_feat_relevance_test.png",
            "IN_best_dec10_edge_rel_3d_test.png"
            )
    else:
        pipeline(graph,features,
            "data/model/IN_best_dec10.pth",
            "IN_best_dec10_node_feat_relevance_j0.png",
            "IN_best_dec10_edge_rel_3d_j0.png"
            )

    if "node_feat_rel" in targets:
        # make feature relevance heatmap at each node
        plot_node_feat_heatmap(r,features,"IN_best_dec10_node_feat_relevance_j0.png")

    if "edge_rel_3d" in targets:
        # process relevance scores for making 3D jet plot
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
        plot_network_3D(G,45,graph.y.detach(),fn=nw_rel_fn,save=True)

    