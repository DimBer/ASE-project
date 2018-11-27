#ifndef CSR_HANDLING_H_
#define CSR_HANDLING_H_

#include "emb_defs.h"

void make_CSR_col_stoch(csr_graph*);

void csr_normalize(csr_graph );

void csr_add_diagonal(csr_graph , double );

void csr_scale(csr_graph, double );

csr_graph csr_create( const sz_long** , sz_long);

csr_graph csr_deep_copy_and_scale(csr_graph, double );

csr_graph* csr_mult_deep_copy( csr_graph, sz_short );

csr_graph csr_deep_copy(csr_graph);

void my_CSR_matmat( double* Y ,double* X  , csr_graph , sz_med , sz_med , sz_med);

void my_CSR_matvec( double* ,double* ,csr_graph);

#endif
