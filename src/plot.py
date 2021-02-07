import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx


sns.set(style="white")

def plot_node_feat_heatmap(r,features,fn="node_feat_heatmap.png",save=True,display=False):
    """
    @param r: a tensor encoding the values to be plotted
    """
    # prepare the values for plotting
    val=r.detach().cpu().numpy()
    df=pd.DataFrame(val,columns=features).T

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8,8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    s=sns.heatmap(df, cmap=cmap, center=0,yticklabels=1,
                square=True, linewidths=0.01, cbar_kws={"shrink": .5})

    s.tick_params(labelsize=7)

    plt.title("source node feature relevance")

    if display:
        plt.show()
    
    if save:
        plt.savefig(fn,bbox_inches='tight')


def plot_network_3D(G, angle,label,fn="edge_rel.png",save=True,display=False):
    """
    @param G: networkx graph to be plotted
    @param angle: view angle of the 3D plot
    @param label: label to be used in the plot title
    @param fn: file name to save the figure if `save` is True
    @param save: whether to save the plot to file
    """
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    node_size=nx.get_node_attributes(G,"node_size")
    
    # Get number of nodes
    n = G.number_of_nodes()

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, c='k',s=25*node_size[key], edgecolors='orange', alpha=0.5)
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        edge_alpha=list(nx.get_edge_attributes(G, 'edge_alpha').values())
        for i,j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
            # plot the edges with specified transparency
            ax.plot(x, y, z, c='b', alpha=edge_alpha[i])
    

    ax.view_init(30, angle)
    plt.xlabel("track_etarel")
    plt.ylabel("track_phirel")
    plt.title(f"jet label: {label}")
    
    if display:
        plt.show()

    if save:
        plt.savefig(fn)
    