# EBGCN
7404 project
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
