///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains PDF program that implements personalized diffusions for recommendations.

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

#include "emb_IO.h"
#include "emb_defs.h"
#include "emb_mem.h"
#include "split_gen.h"
#include "emb_fit.h"
#include "train.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//MAIN
int main( int argc, char **argv )
{
	//Parse arguments using argument parser
	cmd_args args;
	parse_commandline_args(argc,argv,&args);

	// Read adjacency matrix
	printf("Loading graph..\n\n");
	csr_graph graph;
	if(!args.test){
	graph = csr_from_edgelist_file(args.graph_file);
	}else{
	graph = csr_from_edgelist_file(TEST_GRAPH);}

	//Generate graph splits
	printf("Generating graph splits..\n\n");
	edge_tuple** pos_edges = edge_tuple_alloc(args.splits, args.edges_per_split);
	edge_tuple** neg_edges = edge_tuple_alloc(args.splits, args.edges_per_split);

	srand(time(NULL)); //seed the random number generator

	csr_graph* graph_samples = generate_graph_splits(graph, pos_edges, neg_edges, args);

	d_mat thetas = d_mat_init( args.splits, args.walk_length);

	//Fit embedding coefficients
	printf("\nFitting embedding coefficients..\n\n");
	fit_emb_coefficients(graph_samples, &thetas, pos_edges, neg_edges, args);

	//Build embeddings
	printf("\nBuilding embeddings..\n\n");
	d_mat embeddings = d_mat_init(graph.num_nodes, args.dimension);
	build_embeddings(embeddings, graph, thetas, args);

	//Save embeddins
	printf("\nSaving to file..\n\n");
	write_d_mat(embeddings, args.outfile);

    //Free memory
    csr_free(graph);
	csr_array_free(graph_samples, args.splits);
	edge_tuple_free(pos_edges, args.splits);
	edge_tuple_free(neg_edges, args.splits);
	d_mat_free(thetas);
	d_mat_free(embeddings);
	return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
