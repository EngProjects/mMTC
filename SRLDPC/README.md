# Sparse Regression LDPC Code (SR-LDPC) Files

This folder contains all files related to SRLDPC codes and single-user SPARCs. 

### Repository Organization
- Folders:
    - __ldpc_graphs__: contains (modified) alist definition of various LDPC codes
- Files:
    - __benchmarks.py__: Sionna implementation of 5G-NR LDPC codes + 4PAM for benchmarking purposes
    - __gfldpc.py__: GF(2^j) LDPC encoder and BP decoder
    - __srldpc_ber.py__: SR-LDPC BER simulation file
    - __gfldpc_se.py__: utility for MSE message passing
    - __srldpc_approximate_state_evolution.py__: SR-LDPC approximate SE simulations
    - __tau2_to_el0_v4.txt__: LUT of $\tau^2 \leftrightarrow \mathbb{E}\left[\alpha(0)\right]$ values
