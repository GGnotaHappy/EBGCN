# EBGCN 7404 project

Usage process
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




