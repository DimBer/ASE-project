////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains high-level routines for parsing the command line for arguments,
 and handling multiclass or multilabel input.


 Dimitris Berberidis
 University of Minnesota 2017-2018
*/

///////////////////////////////////////////////////////////////////////////////////////////////


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <getopt.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "emb_defs.h"
#include "emb_mem.h"

//Alocate memory for csr
void csr_alloc(csr_graph* graph){
	sz_long nnz_buff = EDGE_BUFF_SIZE , num_nodes_buff = NODE_BUFF_SIZE;

	graph->csr_value = (double*) malloc(nnz_buff*sizeof(double));
	graph->csr_column = (sz_long*) malloc(nnz_buff*sizeof(sz_long));
	graph->csr_row_pointer = (sz_long*) malloc(num_nodes_buff*sizeof(sz_long));
	graph->degrees = (sz_long*) malloc(num_nodes_buff*sizeof(sz_long));
}

//Reallocate csr memory after size is known
void csr_realloc(csr_graph* graph, sz_long nnz, sz_long num_nodes ){
	graph->num_nodes = num_nodes;
	graph->nnz = nnz;
	graph->csr_value = realloc(graph->csr_value, nnz *sizeof(double));
	graph->csr_column = realloc(graph->csr_column, nnz *sizeof(sz_long));
	graph->csr_row_pointer = realloc(graph->csr_row_pointer, (num_nodes+1)*sizeof(sz_long));
	graph->degrees = realloc(graph->degrees, num_nodes*sizeof(sz_long));
}

// Free memory allocated to csr_graph
void csr_free( csr_graph graph ){
	free(graph.csr_value);
	free(graph.csr_column);
	free(graph.csr_row_pointer);
	free(graph.degrees);
}

// Free memory allocated to array of csr_graphs
void csr_array_free(csr_graph* graph_array, sz_short num_copies){
	for(sz_short i=0;i<num_copies;i++) csr_free(graph_array[i]);
	free(graph_array);
}

// Allocate double matrix
d_mat d_mat_init( sz_long num_rows, sz_long num_cols ){
	d_mat mat = {.num_rows=num_rows, .num_cols = num_cols };
	mat.val = (double**) malloc( num_rows*sizeof(double*));
	for(sz_long i=0; i<num_rows; i++) mat.val[i] = (double*) malloc(num_cols*sizeof(double));

	for(sz_long i=0; i<num_rows; i++){
		for(sz_long j=0; j<num_cols; j++) mat.val[i][j]=0.0;
	}

	return mat;
}

//Free double rating_matrix_file
void d_mat_free(d_mat mat){
	for(sz_long i=0; i<mat.num_rows; i++) free(mat.val[i]);
	free(mat.val);
}

// Allocate double vector
d_vec d_vec_init( sz_long num_entries ){
	d_vec vec = {.num_entries = num_entries };
	vec.val = (double*) malloc( num_entries*sizeof(double));
	for(sz_long i=0; i<num_entries; i++) vec.val[i]=0.0;
	return vec;
}

//Free double vector
void d_vec_free(d_vec vec){
	free(vec.val);
}

//Allocate matrix of edge_tuples
edge_tuple** edge_tuple_alloc(sz_med num_rows, sz_med num_cols){
	edge_tuple** edges = (edge_tuple**) malloc(num_rows*sizeof(edge_tuple*));
	for(sz_med i=0;i<num_rows;i++) edges[i] = (edge_tuple*) malloc(num_cols*sizeof(edge_tuple));
	return edges;
}

//free matrix of edge_tuples
void edge_tuple_free(edge_tuple** edges, sz_med num_rows){
	for(sz_med i=0;i<num_rows;i++) free(edges[i]);
	free(edges);
}

//Allocate matrix of sz_long type
sz_long** long_mat_alloc(sz_med num_rows, sz_med num_cols){
	sz_long** long_mat = (sz_long**) malloc(num_rows*sizeof(sz_long*));
	for(sz_med i=0;i<num_rows;i++) long_mat[i] = (sz_long*) malloc(num_cols*sizeof(sz_long));
	return long_mat;
}

//free matrix of sz_long type
void long_mat_free(sz_long** long_mat, sz_med num_rows){
	for(sz_med i=0;i<num_rows;i++) free(long_mat[i]);
	free(long_mat);
}
