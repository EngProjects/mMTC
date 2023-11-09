# Coded Compressed Sensing (CCS) Files

This folder contains all files related to:

- *Coded Demixing for Unsourced Random Access* (https://ieeexplore.ieee.org/document/9795083)
- *Multi-Class Unsourced Random Access via Coded Demixing* (https://ieeexplore.ieee.org/document/9517816)
- *Stochastic Binning and Coded Demixing for Unsourced Random Access* (https://ieeexplore.ieee.org/document/9593113)
- *A Hybrid Approach to Coded Compressed Sensing Where Coupling Takes Place Via the Outer Code* (https://ieeexplore.ieee.org/document/9414469)

### Repository Organization
- Files:
    - **ccsfg.py**: Contains the necessary building blocks to implement a bipartite factor graph tailored to belief propagation.
    - **FactorGraphGeneration.py`: Contains several pre-built factor graphs for use in connection with `ccsfg.py`. 
    - **ccsinnercode.py**: A self-contained compressed sensing encoder/decoder using approximate message passing (AMP) that allows for dynamic information sharing between inner and outer codes. 
    - **ccssimulation.py**: A simple example of how to use `ccsfg.py`, `FactorGraphGeneration.py`, and `ccsinnercode.py` to construct a basic CCS-AMP simulation. 
    - **coded_demixing.py**: Code used in *Coded Demixing for Unsourced Random Access* (https://ieeexplore.ieee.org/document/9795083).
    - **isit2021.py**: Code used in *Multi-Class Unsourced Random Access via Coded Demixing* (https://ieeexplore.ieee.org/document/9517816).
    - **spawc2021.py**: Code used in *Stochastic Binning and Coded Demixing for Unsourced Random Access* (https://ieeexplore.ieee.org/document/9593113).
    - **icassp2021_treecode.ipynb**: Code used in *A Hybrid Approach to Coded Compressed Sensing Where Coupling Takes Place Via the Outer Code* (https://ieeexplore.ieee.org/document/9414469). The companion file `icassp2021_factorgraph.py` offers an alternative implementation of the same functionality. 
