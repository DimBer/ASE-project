#ifndef REC_IO_H_
#define REC_IO_H_

#include "emb_defs.h"

void parse_commandline_args(int ,char**  , cmd_args* );

csr_graph csr_from_edgelist_file( char* );

void write_d_mat(d_mat , char* );

void print_adj(csr_graph );

void print_array(double* , int , int );

#endif
