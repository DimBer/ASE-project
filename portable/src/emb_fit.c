///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains routine to fit embedding coefficients based on fitting missing edges

 Dimitris Berberidis
 University of Minnesota 2017-2018
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <sys/sysinfo.h>
#include <cblas.h>

#include "emb_defs.h"
#include "emb_mem.h"
#include "csr_handling.h"
#include "my_svd.h"
#include "emb_fit.h"
#include "emb_IO.h"
#include "train.h"

static void fit_emb_coefficients_core(csr_graph, d_mat*, d_vec*, double*, edge_tuple*, edge_tuple*, cmd_args);
static d_vec pool(d_mat );
static d_vec majority_vote(d_mat );
static d_vec pool(d_mat );


//List of possible edge fitting methods
void (*fit_edges[NUM_METHODS])(d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                               edge_tuple* neg_edges, double* theta, cmd_args args ) = { logistic_fit,
                                                                                         ls_fit,
                                                                                         svm_fit,
                                                                                         single_best };


//Input: graph, embedding dimensions, and coefficients theta
//Output: node embeddings
void build_embeddings(d_mat embeddings, csr_graph graph, d_mat thetas, cmd_args args){

  //Pool thetas
  d_vec theta;
  if(args.which_fit == 3)
    theta = majority_vote(thetas);
  else
    theta = pool(thetas);

  // Scale to avoid problems with sklearn libraries
  for(int k=0; k<theta.num_entries; k++)
    theta.val[k] *= (double) SCALE;

  //Normalize graph
  csr_normalize(graph);

  //Add identity to adjacency matrix
  csr_add_diagonal(graph, 1.0);

  //Scale down to bring spectral norm to 1
  csr_scale(graph, 0.5);

  //intitialize eigenpairs
  d_mat eigvecs = d_mat_init((sz_long) args.dimension, graph.num_nodes);
  d_vec eigvals = d_vec_init((sz_long) args.dimension);

  //Perform SVD on resulting graph
  my_svd(graph, &eigvecs, &eigvals, args.dimension);

  //Compute node_embeddings
  double S[args.dimension];
  for(sz_med j=0; j<args.dimension; j++) S[j] = 0.0f;
  for(sz_med k = 0; k<args.walk_length; k++){
    for(sz_med j=0; j<args.dimension; j++)
      S[j] += theta.val[k]*pow(eigvals.val[j], k+1);
  }

  for(sz_long i = 0; i<embeddings.num_rows; i++){
    for(sz_med j=0; j<args.dimension; j++)
      embeddings.val[i][j] = S[j] * eigvecs.val[j][i];
  }

  //free
  d_vec_free(theta);
  d_mat_free(eigvecs);
  d_vec_free(eigvals);
}

//Function that fits embedding coefficients to an array of graphs and edge splits
void  fit_emb_coefficients(csr_graph* graph, d_mat* thetas, edge_tuple** pos_edges,
                           edge_tuple** neg_edges, cmd_args args)
{
      //intitialize eigenpairs
      d_mat eigvecs = d_mat_init((sz_long) args.dimension, graph[0].num_nodes);
    	d_vec eigvals = d_vec_init((sz_long) args.dimension);

      for(sz_med i=0; i<args.splits; i++)
        fit_emb_coefficients_core(graph[i], &eigvecs, &eigvals, thetas->val[i],
                                  pos_edges[i], neg_edges[i], args);

      //free eigenpairs
      d_mat_free(eigvecs);
    	d_vec_free(eigvals);
}

//Function that actually fits the coefficients to a given graph and edge split
static void fit_emb_coefficients_core(csr_graph graph, d_mat* eigvecs, d_vec* eigvals, double* theta,
                                      edge_tuple* pos_edges, edge_tuple* neg_edges, cmd_args args)
{
      if(args.test){
        printf("\n\nEdges removed: \n");
        for(sz_long i=0; i<args.edges_per_split; i++)
          printf("%"PRIu64" %"PRIu64"\n",pos_edges[i].node_a,pos_edges[i].node_b);
        printf("\n\n");
      }

      //Normalize graph
      csr_normalize(graph);

      //Add identity to adjacency matrix
      csr_add_diagonal(graph, 1.0);

      //Scale down to bring spectral norm to 1
      csr_scale(graph, 0.5);

      if(args.test) print_adj(graph);

    	//Perform SVD on resulting graph
    	my_svd(graph, eigvecs, eigvals, args.dimension);

      (*fit_edges[args.which_fit])( eigvecs, eigvals, pos_edges, neg_edges, theta, args);

      //Restore graph copy in case it needs to be reused for another split
      for(sz_long i = 0;i<graph.nnz; i++) graph.csr_value[i] = 1.0f;

}


//Aggregates rows of input matrix X to vector y
static d_vec pool(d_mat X){

    d_vec y = d_vec_init(X.num_cols);

    for(sz_med i = 0; i<X.num_cols; i++){
      for(sz_med j = 0; j<X.num_rows; j++)
        y.val[i] += X.val[j][i];
      y.val[i] /= (double) X.num_rows;
    }

    return y;
}

// Decide best step k by majority voting 
// All others are set to 0 and the best is set to SCALE
static d_vec majority_vote(d_mat X){

    d_vec y = d_vec_init(X.num_cols);

    int best_k = 0;
    double best = 0.0f;
    for(sz_med i=0;i<X.num_cols;i++){
      for(sz_med j=0;j<X.num_rows;j++)
        y.val[i] += X.val[j][i];
      if(y.val[i] > best){
        best_k = i;
        best = y.val[i];
      }  
    }

    for(sz_med i=0;i<X.num_cols;i++)
      y.val[i] = 0.0f;
    y.val[best_k] = 1.0;  

    return y;
}




