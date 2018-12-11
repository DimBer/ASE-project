__DESCRIPTION__

Programm __EMB_min__ generates Adaptive-Similarity Node embeddings.
	

__INPUT FILES FORMAT__

__EMB_min__ loads the graph in __edge list__ format from a `.txt` file that contains edges as tab separated pairs of node indexes in the format: `node1_index \tab node2_index`. Node indexes should be in range `[1 , 2^64 ]`. Make sure nodes are enumerated from 1 to N (you can use ``GPT`` for this [here](https://github.com/DimBer/GPT_lib)). 

__OUTPUT FILE FORMAT__

Output is text file containing embeddings. The `i`-th line contains space-separated embedding vector of the `i`-th node.

__COMPILATION__ (LINUX)

Dependencies: `cblas`  must be installed

Command line: `make clean` and then `make`

__EXECUTION__
		      	 
Command line: `./EMB_min [OPTIONS]`

__OPTIONS__

Command line optional arguments with values:

ARGUMENT | TYPE | DEFAULT VALUE
-------- | ------ | -------
`--graph_file` | path to file containing edge-list (see above) | `"../../../../graphs/HomoSapiens/adj.txt"`
`--outfile` | (embeddings) | `"../../embeddings/HomoSapiens_embed.txt"`
`--dimension` | Dimension of embedding space (vector length) | `100`
`--walk_length` | Maximum length of walks (similarity powers) considered | `10`
`--lambda` | l-2 regularization parameter | `1.0e-3`
`--splits` | Number of splits | `2`
`--edges_per_split` | Number of positive edges removed at every split | `1000`     
`--fit_with` | Select model according to which proximity parameters are learnt: 1) Logistic regression (`logistic`), 2) Least squares (`ls`), 3) SVMs `svm`, 4) Chose single best proximity/walk-length (`single best`)  | `single_best`

Command line optional arguments without values:

ARGUMENT | RESULT
-------- | ------
`--simplex` | Constrains parameters to simplex (`TRUE` by DEFAULT)
`--single_thread` | Forces single thread execution ( SVD is single threaded anyway in this implementation)















