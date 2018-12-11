__DESCRIPTION__

Programm `EMB_slepc` generates __Adaptive-Similarity Node Embeddings__ (see [here](https://arxiv.org/pdf/1811.10797.pdf)). 
	

__INPUT FILES FORMAT__

__EMB_min__ loads the graph in __edge list__ format from a `.txt` file that contains edges as tab separated pairs of node indexes in the format: `node1_index \tab node2_index`. Node indexes should be in range `[1 , 2^64 ]`. Make sure nodes are enumerated from 1 to N (you can use ``GPT`` for this [here](https://github.com/DimBer/GPT_lib)). 

__OUTPUT FILE FORMAT__

Output is text file containing embeddings. The `i`-th line contains space-separated embedding vector of the `i`-th node.

__COMPILATION__ (LINUX)

Requires installation of SLEPc package (see [here](http://slepc.upv.es/))

Command line:  `make`

__EXECUTION__

To run (for example) `EMB_slepc` on 8 processes type in command line: `mpiexec -n 8 ./EMB_slepc`

__OPTIONS__

Since command line arguments are "taken" by SLEPc/PETSc, this implementation
uses a configuration file (`config_input`) that can be edited in order to pass arguments to `EMB_slepc`. 
See `portable/` for description of arguments. 













