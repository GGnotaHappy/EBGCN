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

