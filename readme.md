__OVERVIEW__

Implementations of the __Adaptive-similarity Node Embedding__ method  (see [here](https://arxiv.org/pdf/1811.10797.pdf)) 

1) ``portable/`` : Basic implementation with no dependencies (expept basic cblas routines). 

2) ``slepc_based/``: Scalable to larger graphs (millions of nodes). Builds on (and requires installation of) the ``slepc`` package.