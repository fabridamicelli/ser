# SER model on graphs
This minimal model of spreading excitations has a rich history in many disciplines, ranging from the propagation of forest-fires, the spread of epidemics, to neuronal dynamics.
SER stands for susceptible, excited and refractory.

# Installation
```bash
pip install ser
```

# Requirements
 - numpy
 - numba==0.49.1 (other versions might work, but this is the one I tested so far).

Tested in Ubuntu 18.05 with Python 3.8.5.

# Implementation
The graph (or network) is represented as an adjacency matrix (numpy array).
Dynamics is implemented on numba, so it is fast - quick benchmarks show between 2-3 times faster simulations than pure vectorized numpy versions!

# Numba tips and tricks
- Don't use adj_mat with type other than np.float32, np.float64.
- Pro-tip: use np.float32 for adj_mat – it will run faster.

# References
  - J. M. Greenberg and S. P. Hastings, SIAM J. Appl. Math. 34, 515 (1978).
  - A. Haimovici et al. Phys. Rev. Lett. 110, 178101 (2013).
  - Messé et al. PLoS computational biology (2018)

# TODO
  - Tests
  - Implement multi runs
  - Optional turn off numba
