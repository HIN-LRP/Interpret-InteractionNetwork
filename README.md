## Interpreting Higgs Interaction Network with Layerwise Relevance Propagation

#### Abstract

While Graph Interaction Networks achieve exceptional results in Higgs particle identification, GNN Explainer methodology is still in its infancy. We seek to apply Layerwise Relevance Propagation to our existing particle Interaction Network to reveal what features, nodes, and connections are most influential in prediction. 

<hr>

## Description of contents


* `config`: 

* `notebooks`:
    * `EDA.ipynb`: notebook that contains the eploratory data analysis of this project
* `references`: contains links to the paper and libraries referenced in this paper
* `src`:

* `run.py`: Entry point for running different targets of this project
* `runTest.py`: Not intended for direct use, contains script for running the project on Dev data
* `test`: directory for storing Dev data
    * `testdata`
        * `train`: contains Dev data for training
        * `test`: contains Dev data for testing

<hr>

## Build Environment
* [Docker image](https://hub.docker.com/repository/docker/shiro0x19a/higgs-interaction-network) used for this project

<hr>


## Usage
To use `run.py`, a list of supported arguments are provided below
|arguments|purpose|
|-|-|
|`test`| build all targets of this project on Dev data|
|`all`|similar to `test`, but builds with actual data * not recommended as this may take a very long time to finish running|
```
python run.py <arguments>
```
<br>

