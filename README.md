## Interpreting Higgs Interaction Network with Layerwise Relevance Propagation

#### Abstract

While Graph Interaction Networks achieve exceptional results in Higgs particle identification, GNN Explainer methodology is still in its infancy. We seek to apply Layerwise Relevance Propagation to our existing particle Interaction Network to reveal what features, nodes, and connections are most influential in prediction. 

<hr>

## Description of contents


* `notebooks`:
    * `relevance_heatmap.ipynb`: notebook that contains example plots of this project
* `src`:
    * `model`
        * `GraphDataset.py`
        * `InteractionNetwork.py`
    * `sanity_check`
        * `make_data.py`
    * `util`: utility functions such as I/O or plotting
        * `copy.py`
        * `data_io.py`
        * `model_io.py`
        * `plot.py`
    * `LRP.py`: core component of this project

* `run.py`: Entry point for running different targets of this project
* `test`: directory for storing Dev data
    * `test.root`: the generated root file for testing purpose
* `data`
    * `model`: contains a trained IN state dictionary to start with
    * `definitions.yml`: contains metadata definition of the data used in this project
<hr>

## Build Environment
* [Docker image](https://hub.docker.com/repository/docker/shiro0x19a/higgs-interaction-network) used for this project

<hr>


## Usage
To use `run.py`, a list of supported arguments are provided below

For sanity check of the explanation,
```
python run.py sc <arguments>
```
|arguments|purpose|
|-|-|
|`all`|build all targets, equivalent to using [`data` `train` `plot`] as argument|
|`data`| generate dummy data for sanity check|
|`train`| train a dummy IN on the sythesized data|
|`plot`|create static heatmap plots of precomptued relevance scores|



For explaining a pre-trained Higgs boson Interaction Network,
```
python run.py <arguments>
```
|arguments|purpose|
|-|-|
|`test`| build all targets, equivalent to using [`data` `train` `plot`] as argument on Dev data|
|`all`| similar to `test`, but build on actual data|
|`explain`| generate relevance score for given data|
|`plot`| create static heatmap plots of precomptued relevance scores|

<br>

