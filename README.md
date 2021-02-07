## Interpreting Higgs Interaction Network with Layerwise Relevance Propagation

#### Abstract

While Graph Interaction Networks achieve exceptional results in Higgs particle identification, GNN Explainer methodology is still in its infancy. We seek to apply Layerwise Relevance Propagation to our existing particle Interaction Network to reveal what features, nodes, and connections are most influential in prediction. 

<hr>

## Description of contents


* `config`: 

* `notebooks`:
    * `LRP.ipynb`: notebook that contains a demonstration of the content in `IN_LRP.py`
* `references`: contains links to the paper and libraries referenced in this paper
* `src`:
    * `util.py`: contains helper functions
    * `plot.py`: methods for creating the visualizations
    * `IN_LRP.py`: methods for computing relevance score of input using uniform LRP-epsilon rule
* `run.py`: Entry point for running different targets of this project
* `test`: directory for storing Dev data

<hr>

## Build Environment
* [Docker image](https://hub.docker.com/repository/docker/shiro0x19a/higgs-interaction-network) used for this project

<hr>


## Usage
To use `run.py`, a list of supported arguments are provided below
|arguments|purpose|
|-|-|
|`test`| build all targets of this project on Dev data|
|`node_feat_rel`|creates the visualization of node feature relevance score heatmap|
|`edge_rel_3d`|creates the visualization of edge importance of the jet in 3D|
|`all` (or leaving blank)|creates all of the plots mentioned above|
```
python run.py <arguments>
```
<br>

