# Project: Massive Machine Type Communication

This Git repository pertains to massive machine-type communication (mMTC), with a focus on uncoordinated/unsourced random access (URA).
The repository features three branches:
  * **main**: Generic information about this project, along with related resources
  * **github.io**: Source code for [mMTC](https://engprojects.github.io/mMTC/) website
  * **code**: Source code for simulations


## Branch **code**:

This branch contains the source code used to generate published simulation results. This branch is organized as follows: 

- **CCS**: 
  - `ccsfg.py`: Contains the necessary building blocks to implement a bipartite factor graph tailored to belief propagation.
  - `FactorGraphGeneration.py`: Contains several pre-built factor graphs for use in connection with `ccsfg.py`. 
  - `ccsinnercode.py`: A self-contained compressed sensing encoder/decoder using approximate message passing (AMP) that allows for dynamic information sharing between inner and outer codes. 
  - `ccssimulation.py`: A simple example of how to use `ccsfg.py`, `FactorGraphGeneration.py`, and `ccsinnercode.py` to construct a basic CCS-AMP simulation. 
  - `coded_demixing.py`: Code used in *Coded Demixing for Unsourced Random Access* (https://arxiv.org/abs/2203.00239).
  - `isit2021.py`: Code used in *Multi-Class Unsourced Random Access via Coded Demixing* (https://ieeexplore.ieee.org/document/9517816).
  - `spawc2021.py`: Code used in *Stochastic Binning and Coded Demixing for Unsourced Random Access* (https://ieeexplore.ieee.org/document/9593113).
  - `icassp2021_treecode.ipynb`: Code used in *A Hybrid Approach to Coded Compressed Sensing Where Coupling Takes Place Via the Outer Code* (https://ieeexplore.ieee.org/document/9414469). The companion file `icassp2021_factorgraph.py` offers an alternative implementation of the same functionality. 
- **FASURA**:
  - Contains code used in *FASURA: A Scheme for Quasi-Static Massive MIMO Unsourced Random Access Channels* (https://ieeexplore.ieee.org/abstract/document/9833940).
- **HybridUIR**:
  - `hashbeam.ipynb`: Code used in *HashBeam: Enabling Feedback Through Downlink Beamforming in Unsourced Random Access* (https://arxiv.org/abs/2206.01684). 
- **SRLDPC**:
  - `gfldpc.py`: Contains code for encoding and decoding non-binary LDPC codes
  - `srldpc.py`: Code used to obtain results in *Sparse Regression LDPC Codes*