#ifndef REC_MEM_H_
#define REC_MEM_H_

#include "emb_defs.h"

void csr_alloc(csr_graph* );

void csr_realloc(csr_graph* , sz_long , sz_long );

void csr_free( csr_graph );

void csr_array_free( csr_graph* , sz_short);

d_mat d_mat_init( sz_long , sz_long );

void d_mat_free(d_mat );

d_vec d_vec_init( sz_long );

void d_vec_free(d_vec);

edge_tuple** edge_tuple_alloc(sz_med, sz_med );

void edge_tuple_free(edge_tuple** , sz_med );

sz_long** long_mat_alloc(sz_med, sz_med );

void long_mat_free(sz_long** , sz_med );

#endif
