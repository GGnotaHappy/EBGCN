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






