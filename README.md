# DeFoG: Extended Evaluation, Replication, and Improvements

## Introduction

This repository is based on the original **DeFoG** framework, available at:  
https://github.com/manuelmlmadeira/DeFoG

DeFoG provides a generative model for graph synthesis using discrete normalizing flows.  
In this project, we first focused on **replicating the experimental results** from the original paper and validating the behavior of the provided checkpoints. After achieving reproducibility, we explored several **possible improvements to the sampling process**, as mentioned in the report.

This repository includes all the scripts and utilities required to **run every experiment performed in the project**, so that any user can fully reproduce the analysis and results described in the report.

The following sections explain how to run the experiments and how the repository is organized.

---

## Repository Structure
```
├── external/
├── source/
└── scripts/
```

### `external/`
This folder contains the full DeFoG source code, including all modifications introduced during the project.  
These changes include new sampling options, additional distortion modes, hyperparameter hooks, visualization updates, and support for automated experiment pipelines.

### `source/`
This folder contains helper modules used to orchestrate experiments without duplicating code.  
It includes:

- Hyperparameter search utilities (grid and random search)
- Parameter prediction using the MLP model
- Experiment execution wrappers
- Utility functions for evaluation and batch processing

### `scripts/`
This directory contains all executable scripts used during the project to run individual experiments.  
Each script corresponds to a specific test, such as:

- Replication of original results
- Hyperparameter searches
- Base-vs-MLP parameter comparison
- Dataset-specific analyses (planar, tree, sbm, qm9, tls…)
- ...

These scripts demonstrate how to use the tools inside `source/` to reproduce every performed experiment.

### Required Checkpoints

To run any experiment, create a folder:

```
└── checkpoints/
```

and add the necessary model checkpoints, available online. The paper checkpoints are [available here](https://drive.switch.ch/index.php/s/MG7y2EZoithAywE). While the new checkpoints that we trained and uploaded are [available here](https://drive.google.com/drive/folders/1jbasupK7soaOprirxt9ay4c-65Sw154b?usp=sharing).

To reproduce **all** experiments in the project, the required files are:

- `planar.ckpt`
- `tree.ckpt`
- `sbm.ckpt`
- `planar-59999-14h.ckpt`
- `tree-19999-5h.ckpt`
- `sbm-31999-40h.ckpt`
- `qm9_with_h_conditional.ckpt`
- `qm9_no_h.ckpt`


---

## Results Reproduction

This section explains how to **replicate the experiments** from the original DeFoG paper using the scripts in this repository.  
All experiments can be executed in the provided scripts, and outputs are saved in organized directories.


The replication is divided into three main sets of experiments:

1. **Table 1: Performance Metrics**  
    Scripts and functions to reproduce the evaluation of generated graphs across datasets and sampling steps.
    First, you need to run `run_table1.py` by:
    ```
    python -m scripts.run_table1 {version}
    ```
    - version is a parameter that could be 'new' or 'old' based on the checkpoints you would like to use. 

    This runs all the necessary experiments and saves all the data on its respective folders. So then, you would need to run the script `make_table1.py` in order to create a csv file that contains the proper table 1 in a single file by:
    ```
    python -m scripts.make_table1 {version}
    ```
    - version represents the same as before.

2. **Figure 2a: Evolution of Generation Quality**  
    Scripts to run experiments measuring generation quality as a function of the number of sampling steps.  
    For planar graphs, V.U.N. metrics are computed; for QM9 molecules, the validity metric is tracked.
    First, you need to run `run_fig2a.py` by:
    ```
    python -m scripts.run_fig2a --dataset={dataset}
    ```
    - dataset is a parameter that specifies the dataset you use (planar for the left plot of the figure and qm9 for the right one).
    There are other parameters that you could modify, but for replicating exactly figure 2a leave the default values. This would run all the experiments necessary to get all the data. Once you have these data, you can run the script `plot_fig2a.py` that would generate the plot that replicates figure 2a by:
    ```
    python -m scripts.plot_fig2a {dataset}
    ```
    - dataset is again the same as before
    You should run both commands with dataset=planar and dataset=qm9 in order to get both plots of the figure.


3. **Figure 3: Impact of Sampling Configurations**  
    Scripts to explore how different sampling configurations,such as distortion, target guidance, and stochasticity, affect the generated graphs.
    First, you need to run `run_fig3.py` by:
    ```
    python -m scripts.run_fig3
    ```
    This would run all the experiments necessary (modifying the number of steps and the values of eta, omega and distortion types) saving them in thheir respective folders. So, in order to create the actual plots, you would need to run `plot_fig3.py` by:
    ```
    python -m scripts.plot_fig3
    ```
    This would save all the replicated plots included in figure 3 (all 6 of them).

---


## Additional Performed Experiments

Beyond the replication of the original DeFoG results, we conducted a set of **additional experiments** to explore new behaviors, sampling configurations, and model variations.  
Since the structure of these experiments follow the same pattern as those used for result reproduction, we only provide an overview.

The additional experiments include:

- **Distortion Function Variants**  
  Evaluation of multiple distortion functions and their effect on sampling quality and convergence.  
  *(Scripts: `run_distortion_experiments.py`, `plot_distortion_experiments.py`)*

- **Fixed Subgraph Experiments (Fixed Nodes)**  
  Testing the impact of fixing subsets of nodes or subgraphs during generation, including full validity analysis and discussion of structural constraints. In order to run this experiment, you should generate the desired subgraphs to be fixed. For example, we decided to first fix some nodes and edges from original graphs of the validation set, in order to ensure the validity of the select subgraph. Then you would be able to generate constrained graphs that would maintain that subgraph throughout the whole generation process.
  *(Scripts: `run_subgraphs.py`, `make_validity_table.py`, including all tested options explained on the report, `plot_discussion_len_fixed_nodes.py`)*

- **Hyperparameter Search (Grid Search & Random Search)**  
  Automated exploration of hyperparameters through systematic grid searches and stochastic random searches.  
  *(Script: `run_defog_search.py`)*

- **QM9 Experiments and Conditional Generation**  
  Grid search–based tuning for conditional molecular generation, followed by additional analyses on generated energies.  
  *(Scripts: `plot_qm9_cond.py`, `plot_qm9_with_h.py`, `energy_sweeps.py`, `plot_energy_sweeps.py`)*

- **Hyperparameter Optimization Using an MLP Model**  
  Comparison between the default base model and the MLP-based hyperparameter prediction module, evaluating performance across different sampling steps.  
  *(Scripts: `run_default_vs_model.py`, `plot_default_vs_model.py`)*

---

## Modifications to the Original DeFoG Code

As already mentioned, our entire project is based on the **DeFoG paper** and its official implementation.  
To adapt the framework to our experiments and to incorporate new evaluation procedures, sampling strategies, and fixed nodes during sampling, we modified several parts of the original source code.

To make our changes easy to identify, **every addition or correction is enclosed between special hashtag markers (`### OUR CODE START` / `### OUR CODE END`)** across all edited files.  
This allows any user to quickly locate and inspect the exact modifications introduced during the project.

Below is the complete list of files that were modified or added:

### **Modified files**
- `configs/config.yaml`  
- `environment.yaml`  
- `src/analysis/rdkit_functions.py`  
- `src/analysis/visualization.py`
- `src/datasets/qm9_dataset.py`  
- `src/flow_matching/time_distorter.py`  
- `src/graph_discrete_flow_model.py`  
- `src/main.py`  
- `src/utils.py`  

### **New experiment configuration files**
- `configs/experiment/qm9_conditional.yaml`  
- `configs/experiment/qm9_with_h_conditional.yaml`  

All modifications were necessary to support the experimentation for this project.






