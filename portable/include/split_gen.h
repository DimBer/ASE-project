#ifndef SPLIT_GEN_H_
#define SPLIT_GEN_H_

#include "emb_defs.h"

csr_graph*  generate_graph_splits(csr_graph, edge_tuple**, edge_tuple** , cmd_args);

//struct that is passed to each AdaDIF_rec thread
typedef struct {
	csr_graph graph;
  csr_graph* graph_out;
  edge_tuple** pos_edges;
	edge_tuple** neg_edges;
	sz_med num_edges;
  sz_med from_split;
  sz_med local_num_splits;
	bool test;
} split_gen_thread_type;



#endif
