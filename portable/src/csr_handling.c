///////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains routines for handling compressed-sparse-row (CSR) graphs
 ( allocating , copying, normalizing, scaling, mat-vec, mat-mat, freeing )

 Dimitris Berberidis
 University of Minnesota 2017-2018
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>

#include "csr_handling.h"
#include "emb_defs.h"
#include "emb_mem.h"

// Make a copy of graph with edges multiplied by some scalar
csr_graph csr_deep_copy_and_scale(csr_graph graph, double scale ){

	csr_graph graph_temp;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value=(double*)malloc(graph.nnz*sizeof(double));

	graph_temp.csr_column=(sz_long*)malloc(graph.nnz*sizeof(sz_long));

	graph_temp.csr_row_pointer=(sz_long*)malloc((graph.num_nodes+1)*sizeof(sz_long));

	graph_temp.degrees=(sz_long*)malloc(graph.num_nodes*sizeof(sz_long));

	graph_temp.num_nodes=graph.num_nodes;

  graph_temp.nnz=graph.nnz;

  //copy data

  memcpy(graph_temp.csr_row_pointer,graph.csr_row_pointer, (graph.num_nodes+1)*sizeof(sz_long));

  memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes*sizeof(sz_long));

  memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz*sizeof(sz_long));

  for(sz_long i=0;i<graph.nnz;i++){
  	graph_temp.csr_value[i]=scale*graph.csr_value[i];
  }

	return graph_temp;
}

// Make a copy of graph with edges multiplied by some scalar
csr_graph csr_deep_copy(csr_graph graph){

	csr_graph graph_temp;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value=(double*)malloc(graph.nnz*sizeof(double));

	graph_temp.csr_column=(sz_long*)malloc(graph.nnz*sizeof(sz_long));

	graph_temp.csr_row_pointer=(sz_long*)malloc((graph.num_nodes+1)*sizeof(sz_long));

	graph_temp.degrees=(sz_long*)malloc(graph.num_nodes*sizeof(sz_long));

	graph_temp.num_nodes=graph.num_nodes;

  graph_temp.nnz=graph.nnz;

  //copy data

  memcpy(graph_temp.csr_row_pointer,graph.csr_row_pointer, (graph.num_nodes+1)*sizeof(sz_long));

  memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes*sizeof(sz_long));

  memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz*sizeof(sz_long));

  memcpy(graph_temp.csr_value, graph.csr_value, graph.nnz*sizeof(double));

	return graph_temp;
}

//Return an array with multiple copies of the input graph
csr_graph* csr_mult_deep_copy( csr_graph graph, sz_short num_copies ){
	csr_graph* graph_array=(csr_graph*)malloc(num_copies*sizeof(csr_graph));
	for(sz_short i=0;i<num_copies;i++){
		graph_array[i]=csr_deep_copy(graph);
	}
	return graph_array;
}

//Subroutine: modify csr_value to be column stochastic
//First find degrees by summing element of each row
//Then go through values and divide by corresponding degree (only works for undirected graph)
void make_CSR_col_stoch(csr_graph* graph){
	for(sz_long i=0;i<graph->nnz;i++){
		graph->csr_value[i]=graph->csr_value[i]/(double)graph->degrees[graph->csr_column[i]];
	}
}

//Normalization of type D^-1/2 * A * D^-1/2 that retains symmetricity
void csr_normalize(csr_graph graph){

	double* true_degrees = (double*) malloc(graph.num_nodes*sizeof(double));

	for(sz_long i=0;i<graph.num_nodes;i++){
		true_degrees[i]=0.0f;
		for(sz_long j=graph.csr_row_pointer[i];j<graph.csr_row_pointer[i+1];j++){
		 true_degrees[i]+=graph.csr_value[j];
		}
	}

	for(sz_long i=0;i<graph.num_nodes;i++){
		for(sz_long j=graph.csr_row_pointer[i];j<graph.csr_row_pointer[i+1];j++)
			graph.csr_value[j]=graph.csr_value[j]/(double)sqrt(true_degrees[graph.csr_column[j]] * true_degrees[i]);
	}

	free(true_degrees);
}


//Subroutine: take x, multiply with csr matrix from right and store result in y
void my_CSR_matvec( double* y ,double* x  , csr_graph graph){

	for(sz_long i=0;i<graph.num_nodes;i++)
		y[i]=0.0;

	for(sz_long i=0;i<graph.num_nodes;i++)
	{
		for(sz_long j=graph.csr_row_pointer[i];j<graph.csr_row_pointer[i+1];j++)
			y[i]+=x[graph.csr_column[j]]*graph.csr_value[j];
	}
}


//Subroutine: take X, multiply with csr matrix from right and store result in Y
void my_CSR_matmat( double* Y ,double* X  , csr_graph graph, sz_med M, sz_med from, sz_med to){

	for(sz_long i=0;i<graph.num_nodes;i++){for(sz_long j=from;j<to;j++){Y[i*M+j]=0.0f;}}

	for(sz_long i=0;i<graph.num_nodes;i++){
		for(sz_long j=graph.csr_row_pointer[i];j<graph.csr_row_pointer[i+1];j++){
			for(sz_med k=from;k<to;k++){ Y[i*M + k] +=  X[ M*graph.csr_column[j] + k]*graph.csr_value[j];}
		}
	}
}

//Adds  scale*I to graph adjacency matrix
//Only works for csr graphs with preallocated diagonal
void csr_add_diagonal(csr_graph graph, double scale){

	for(sz_long i =0; i<graph.num_nodes; i++){
			bool flag = false;
			for(sz_long j = graph.csr_row_pointer[i]; j<graph.csr_row_pointer[i+1]; j++){
					if(graph.csr_column[j] == i ){
							graph.csr_value[j] = scale;
							flag = true;
							break;
					}
			}
			if(flag == false ) printf("Warning: diagonal element not allocatedi n csr matrix\n");
	}
}

//Scales adjacency matrix

void csr_scale(csr_graph graph, double scale){

	for(sz_long i=0; i<graph.nnz; i++) graph.csr_value[i] *= scale ;

}
