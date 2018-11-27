#ifndef EMB_FIT_H_
#define EMB_FIT_H_

#include "emb_defs.h"

void  fit_emb_coefficients(csr_graph* , d_mat*  , edge_tuple** , edge_tuple**, cmd_args);

void build_embeddings(d_mat, csr_graph, d_mat, cmd_args );

#endif
