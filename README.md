# Project: Massive Machine Type Communication

This Git repository pertains to massive machine-type communication (mMTC), with a focus on uncoordinated/unsourced random access (URA).
The repository features three branches:
  * **main**: Generic information about this project, along with related resources
  * **github.io**: Source code for [mMTC](https://engprojects.github.io/mMTC/) website
  * **code**: Source code for simulations


## Simulation Code

  * **ccsfg.py**: Contains the necessary building blocks to implement a bipartite factor graph tailored to belief propagation.
  * **FactorGraphGeneration.py**: Contains several pre-built factor graphs for use in connection with `ccsfg.py`. 
  * **ccsinnercode.py**: A self-contained compressed sensing encoder/decoder using approximate message passing (AMP) that allows for dynamic information sharing between inner and outer codes. 
  * **CCS-Simulation.py**: A simple example of how to use `ccsfg.py`, `FactorGraphGeneration.py`, and `ccsinnercode.py` to construct a basic CCS-AMP simulation. 
  * **conferenceyear.py**: Code used to generate simulation results for the paper presented to the listed conference in the listed year. 