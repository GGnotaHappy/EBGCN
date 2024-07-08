# Usage process
1. Open the Process file first and run getTwittergraph.py to preprocess the data. Note that two data sets need to be processed, so they need to be run separately. Modify obj in line 136 of the code

```python
   python3 getTwittergraph.py
```

2. Run main_bigcn.py and main_ebgcn.py files separately to train the model

   
```python
   python3 main_bigcn.py
   python3 main_ebgcn.py
```


# Bi-GCN Model Training

This code trains a Bi-Directional Graph Convolutional Network (Bi-GCN) for rumor detection on social media. The model implementation is based on the paper "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks" by Tian Bian et al.

## Requirements

- Python 3.x
- PyTorch
- torch-geometric
- NumPy

## Setup

1. Clone the repository and navigate to the project directory.

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Detailed Script Information
Training Script: The training process is implemented in the train_model function, which initializes the model, optimizer, and other training components. It iterates over the epochs, loading data in batches, calculating loss, performing backpropagation, and evaluating model performance.

Early Stopping: The EarlyStopping class is used to monitor validation loss and stop training when there is no improvement for a specified number of epochs (patience).

Evaluation Metrics: The script calculates various evaluation metrics, including accuracy, precision, recall, and F1 scores for different classes.

## Configuration
The script can be configured using various command-line arguments. Some of the key arguments include:

--datasetname: Name of the dataset (default: "Twitter16").
--input_features: Dimension of input features (TF-IDF) (default: 5000).
--hidden_features: Dimension of graph hidden state (default: 64).
--output_features: Dimension of output features (default: 64).
--num_class: Number of classes (default: 4).
--seed: Random state seed (default: 2020).
--lr: Learning rate (default: 0.0005).
--batchsize: Batch size (default: 128).
--n_epochs: Number of max epochs (default: 200).
--patience: Patience for early stop (default: 10).
--TDdroprate: Drop rate for edges in the top-down propagation graph (default: 0.2).
--BUdroprate: Drop rate for edges in the bottom-up dispersion graph (default: 0.2).
For a full list of arguments, refer to the argparse section in the script.

# EB-GCN Model Training

## Introduction
This script is used to train the EBGCN model for classification tasks on given graph-structured data. The model includes a context-aware bidirectional graph convolutional network.

## File Structure
- `main_ebgcn.py`: Main script containing the primary workflow for training the model.
- `Process/process.py`: Data processing module.
- `tools/earlystopping.py`: Early stopping mechanism module.
- `Process/rand5fold.py`: 5-fold cross-validation module.
- `tools/evaluate.py`: Model evaluation module.
- `model/EBGCN.py`: EBGCN model definition module.

## Dependencies
Ensure the following dependencies are installed:
- Python 3.x
- torch
- torch-geometric
- numpy

## Arguments
--datasetname      str   default: "Twitter16"   Name of the dataset
--modelname        str   default: "BiGCN"       Model type, options: BiGCN/EBGCN
--input_features   int   default: 5000          Dimension of input features (TF-IDF)
--hidden_features  int   default: 64            Dimension of graph hidden state
--output_features  int   default: 64            Dimension of output features
--num_class        int   default: 4             Number of classes
--num_workers      int   default: 30            Number of workers for training

--seed             int   default: 2020          Random seed
--no_cuda          bool  action: 'store_true'   Disable GPU usage
--num_cuda         int   default: 0             GPU index (0/1)

--lr               float default: 0.0005        Learning rate
--lr_scale_bu      int   default: 5             Learning rate scale for bottom-up direction
--lr_scale_td      int   default: 1             Learning rate scale for top-down direction
--l2               float default: 1e-4          L2 regularization weight
--dropout          float default: 0.5           Dropout rate
--patience         int   default: 10            Patience for early stopping
--batchsize        int   default: 128           Batch size
--n_epochs         int   default: 200           Number of max epochs
--iterations       int   default: 50            Number of iterations for 5-fold cross-validation

--TDdroprate       float default: 0.2           Drop rate for edges in top-down graph
--BUdroprate       float default: 0.2           Drop rate for edges in bottom-up graph
--edge_infer_td    bool  action: 'store_true'   Enable edge inference in top-down graph
--edge_infer_bu    bool  action: 'store_true'   Enable edge inference in bottom-up graph
--edge_loss_td     float default: 0.2           Weight for unsupervised relation learning loss in top-down graph
--edge_loss_bu     float default: 0.2           Weight for unsupervised relation learning loss in bottom-up graph
--edge_num         int   default: 2             Latent relation types in edge inference




