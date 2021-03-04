import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bokeh.io import save
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter,)
from bokeh.plotting import figure
from bokeh.transform import transform

sns.set(style="white")

def plot_static(R,ix,features,save_to="jet_0.png"):
    r=(R[ix]["node"]).clone()
    r[torch.isnan(r)]=0

    val=r.detach().cpu().numpy()
    # val=val[sort_pt_idx]
    df=pd.DataFrame(val,columns=features).T

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8,8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    s=sns.heatmap(df, cmap=cmap, center=0,yticklabels=1,
                square=True, linewidths=0.01, cbar_kws={"shrink": .5})

    s.tick_params(labelsize=7)

    
    if R[ix]['label'][:,1]>0:
        title_str="node feature relevance of a signal data sample\n\n"
    else:
        title_str="node feature relevance of a background data sample\n\n"

    plt.title(title_str+"prediction:{}".format(R[ix]["pred"].detach().cpu().numpy().round(4)))
    plt.savefig(save_to)

def plot_interactive(R,ix,raw_input,features,save_to="jet_0.html"):
    r=(R[ix]["node"]).clone()
    r[torch.isnan(r)]=0

    val=r.detach().cpu().numpy()
    data=pd.DataFrame(val,columns=features)
    data.columns.name="feature"
    data.index.name="particle"
    
    df = pd.DataFrame(data.stack(), columns=['relevance']).reset_index()

    raw=raw_input[ix].x
    sort_idx=torch.argsort(raw[:,0])
    raw=raw[sort_idx]
    df["raw data"]=raw.reshape(-1,1).clone().detach().numpy()
    
    df["particle"]=df["particle"].astype(str)
    df.drop_duplicates(['particle','feature'],inplace=True)
    data = data.T.reset_index().drop_duplicates(subset='feature').set_index('feature').T
    source = ColumnDataSource(df)
    
    colors=[]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(matplotlib.colors.rgb2hex(rgba))

    scale=max(np.abs(df.relevance.min()),np.abs(df.relevance.max()))
    mapper = LinearColorMapper(palette=colors, low=-scale, high=scale)

    if R[ix]['label'][:,1]>0:
        title_str="Higg boson jet node relevance, "
    else:
        title_str="QCD jet node relevance, "

    title_str+="prediction:{}".format(R[ix]["pred"].detach().cpu().numpy().round(4)[0])


    p = figure(x_range=[str(i) for i in data.index],
               y_range=list(reversed(data.columns)),
               title=title_str,
               tools=["hover"])

    p.rect(x="particle", y="feature", width=1, height=1, source=source,
           line_color='white', fill_color=transform('relevance', mapper))

    p.hover.tooltips = [
        ("particle", "@particle"),
        ("feature", "@feature"),
        ("relevance score", "@relevance"),
        ("input data","@{raw data}")
    ]

    color_bar = ColorBar(color_mapper=mapper,
                         ticker=BasicTicker(desired_num_ticks=10),
                         location=(0,0),
                         formatter=PrintfTickFormatter(format="%d"))

    p.add_layout(color_bar,'right')
    save(p,save_to)

def plot(R,ix,raw_input,features,save_to=f"jet_0"):
    plot_static(R,ix,features,save_to+".png")
    plot_interactive(R,ix,raw_input,features,save_to+".html")