# README

## Preparation

#### Data
Data is only available on kaggle currently. In order to run example code, you need to download data from kaggle and put into *data* folder in project root. See *examples/pairing.py* for more detail.

### Environment
Please add project root to your $PYTHONPATH or virtual environment package path in order to import package properly. If you are using PyCharm, this should be default behavior.

### Dependencies
- SciKit-learn
- Pandas
- PyTorch
- Matplotlib

## Preprocessing
Data preprocessing is implement in *cpt\_data\_preprocessing*.
See *examples/filters.py* and *examples/pairing.py* for more detail.

## GNN
GNN model is implement in *cpt\_gnn*. Training option can be found in *examples/configs*. See *examples/gnn.py* for more detail.

Note that you should run *examples/pairing.py* first to generate pairing data.

## Visualization
Visualization is implement in *examples/cpt_plots*. It provide many drawing method for different situation. See *examples/gnn.py* and *examples/plot_500_particles.py* for more detail.
