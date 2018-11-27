///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains PDF program that implements personalized diffusions for recommendations.

 Dimitris Berberidis
 University of Minnesota 2017-2018
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "slepcsys.h"
#include "slepcsvd.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <pthread.h>
#include <cblas.h>

#define PRINT_THETAS 1

//Input buffer sizes

#define EDGE_BUFF_SIZE 10000000

#define NODE_BUFF_SIZE 2000000

//Default command line arguments

#define DEFAULT_NUM_WALK 10

#define DEFAULT_DIMENSION 100

#define DEFAULT_SPLITS 2

#define DEFAULT_EDGES_PER_SPLIT 1000

#define DEFAULT_GRAPH "../../../../graphs/HomoSapiens/adj.txt"

#define DEFAULT_OUTFILE "../../embeddings/HomoSapiens_embed.txt"

#define DEFAULT_SINGLE_THREAD true

#define DEFAULT_TEST false

#define DEFAULT_SIMPLEX false

#define DEFAULT_LAMBDA 1.0e-3

#define DEFAULT_FIT 3

#define TEST_GRAPH "../../../../graphs/test.txt"

//Default optimization parameters

#define SCALE 200.0f

#define GD_TOL 1.0e-6

#define GD_TOL_2 1.0e-6

#define STEPSIZE 0.1

#define STEPSIZE_2 0.95

#define STEPSIZE_LOG 0.5

#define STEPSIZE_SVM 0.1

#define MAXIT_GD 10000

#define PROJ_TOL 1.0e-6

#define MAX_SAMPLE 1e7

#define HINGE_EPSILON 1.0e-3

#define DC_THRES 5

#define DC_WIN_LEN 50

#define NUM_METHODS 4

//INTEGER TYPES

typedef uint64_t sz_long_t; //Long range unsinged integer. Used for node and edge indexing.

typedef uint32_t sz_med_t; //Medium range unsinged integer. Used for random walk length,
						   //iteration indexing and seed indexing

typedef uint8_t sz_short_t; //Short range unsigned integer. Used for class and thread indexing.

typedef int8_t class_t; // Short integer for actual label values.

//DATA STRUCTURES

// Record of user-item matrix as an array of user STRUCTURES

typedef struct
{
	sz_long_t node_a;
	sz_long_t node_b;
} edge_tuple_t;

//Double and index struct for sorting and keeping indexes

typedef struct
{
	double val;
	int ind;
} val_and_ind_t;

//struct forcommand line arguments
typedef struct
{
	int *argc;
	char ***argv;
} argc_t;

//struct for parameters
typedef struct
{
	char *graph_file;
	char *outfile;
	sz_med_t walk_length;
	sz_med_t splits;
	sz_med_t edges_per_split;
	sz_med_t dimension;
	double lambda;
	bool single_thread;
	bool test;
	bool simplex;
	sz_short_t which_fit;
	argc_t cmd;
} cmd_args_t;

//Csr graph struct
typedef struct
{
	double *csr_value;
	sz_long_t *csr_column;
	sz_long_t *csr_row_pointer;
	sz_long_t num_nodes;
	sz_long_t nnz;
	sz_long_t *degrees;
} csr_graph_t;

//Double matrix
typedef struct
{
	double **val;
	sz_long_t num_rows;
	sz_long_t num_cols;
} d_mat_t;

//Double vector
typedef struct
{
	double *val;
	sz_long_t num_entries;
} d_vec_t;

static void single_best(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
						edge_tuple_t *neg_edges, double *theta, cmd_args_t args);

static void ls_fit(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
				   edge_tuple_t *neg_edges, double *theta, cmd_args_t args);

static void svm_fit(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
					edge_tuple_t *neg_edges, double *theta, cmd_args_t args);

static void logistic_fit(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
						 edge_tuple_t *neg_edges, double *theta, cmd_args_t args);

static void make_CSR_col_stoch(csr_graph_t *);

static void csr_normalize(csr_graph_t);

static void csr_add_diagonal(csr_graph_t, double);

static void csr_scale(csr_graph_t, double);

static csr_graph_t csr_create(const sz_long_t **, sz_long_t);

static csr_graph_t csr_deep_copy_and_scale(csr_graph_t, double);

static csr_graph_t *csr_mult_deep_copy(csr_graph_t, sz_short_t);

static csr_graph_t csr_deep_copy(csr_graph_t);

static void my_CSR_matmat(double *Y, double *X, csr_graph_t, sz_med_t, sz_med_t, sz_med_t);

static void my_CSR_matvec(double *, double *, csr_graph_t);

static void fit_emb_coefficients(csr_graph_t *, d_mat_t *, edge_tuple_t **, edge_tuple_t **, cmd_args_t);

static void build_embeddings(d_mat_t, csr_graph_t, d_mat_t, cmd_args_t);

static void parse_config_input_file(cmd_args_t *);

static csr_graph_t csr_from_edgelist_file(char *);

static void write_d_mat(d_mat_t, char *);

static void print_adj(csr_graph_t);

static void print_array(double *, int, int);

static void csr_alloc(csr_graph_t *);

static void csr_realloc(csr_graph_t *, sz_long_t, sz_long_t);

static void csr_free(csr_graph_t);

static void csr_array_free(csr_graph_t *, sz_short_t);

static d_mat_t d_mat_init(sz_long_t, sz_long_t);

static void d_mat_free(d_mat_t);

static d_vec_t d_vec_init(sz_long_t);

static void d_vec_free(d_vec_t);

static edge_tuple_t **edge_tuple_alloc(sz_med_t, sz_med_t);

static void edge_tuple_free(edge_tuple_t **, sz_med_t);

static sz_long_t **long_mat_alloc(sz_med_t, sz_med_t);

static void long_mat_free(sz_long_t **, sz_med_t);

static csr_graph_t *generate_graph_splits(csr_graph_t, edge_tuple_t **, edge_tuple_t **, cmd_args_t);

static void my_svd(csr_graph_t, d_mat_t *, d_vec_t *, sz_med_t, argc_t);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*----------------------------------------------------------------------------------------------------------------------*/

//	MAIN  //
/*---------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{

	//Initialize SLEPc
	char help[] = "Initializing SLEPc\n";
	SlepcInitialize(&argc, &argv, (char *)0, help);

	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	printf(" Rank is %d \n", (int)rank);

	//Parse arguments using argument parser
	cmd_args_t args;
	parse_config_input_file(&args);
	args.cmd.argc = &argc;
	args.cmd.argv = &argv;

	// Read adjacency matrix
	if (rank == 0)
		printf("Loading graph..\n\n");
	csr_graph_t graph;
	if (!args.test)
	{
		graph = csr_from_edgelist_file(args.graph_file);
	}
	else
	{
		graph = csr_from_edgelist_file(TEST_GRAPH);
	}

	//Generate graph splits
	if (rank == 0)
		printf("Generating graph splits..\n\n");
	edge_tuple_t **pos_edges = edge_tuple_alloc(args.splits, args.edges_per_split);
	edge_tuple_t **neg_edges = edge_tuple_alloc(args.splits, args.edges_per_split);

	srand(0); //seed the random number generator

	csr_graph_t *graph_samples = generate_graph_splits(graph, pos_edges, neg_edges, args);

	d_mat_t thetas = d_mat_init(args.splits, args.walk_length);

	//Fit embedding coefficients
	if (rank == 0)
		printf("\nFitting embedding coefficients..\n\n");
	fit_emb_coefficients(graph_samples, &thetas, pos_edges, neg_edges, args);

	//Build embeddings
	if (rank == 0)
		printf("\nBuilding embeddings..\n\n");
	d_mat_t embeddings = d_mat_init(graph.num_nodes, args.dimension);
	build_embeddings(embeddings, graph, thetas, args);

	//Save embeddins
	if (rank == 0)
	{
		printf("\nSaving to file..\n\n");
		write_d_mat(embeddings, args.outfile);
	}

	//Free memory
	csr_free(graph);
	csr_array_free(graph_samples, args.splits);
	edge_tuple_free(pos_edges, args.splits);
	edge_tuple_free(neg_edges, args.splits);
	d_mat_free(thetas);
	d_mat_free(embeddings);
	free(args.graph_file);
	free(args.outfile);

	//Finalize SLEPc
	int ierr = SlepcFinalize();
	CHKERRQ(ierr);

	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*----------------------------------------------------------------------------------------------------------------------*/

//	SPARSE SVD  //
/*---------------------------------------------------------------------------------------------------------------------*/

//Wrapper of SLEPC sparse svd
static void my_svd(csr_graph_t graph, d_mat_t *U, d_vec_t *S, sz_med_t dimension, argc_t cmd)
{

	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	sz_long_t average_degree = 0;
	for (sz_long_t i = 0; i < graph.num_nodes; i++)
		average_degree += graph.csr_row_pointer[i + 1] - graph.csr_row_pointer[i];

	average_degree = average_degree / graph.num_nodes;

	if (rank == 0)
		printf("average degree: %d\n", (int)average_degree);

	PetscPrintf(PETSC_COMM_WORLD, "Calling SVD from SLEPc..\n\n");

	Mat A;
	SVD svd;

	MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, (PetscInt)graph.num_nodes, (PetscInt)graph.num_nodes,
				 (PetscInt)average_degree, NULL, (PetscInt)average_degree, NULL, &A);

	MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
	MatSetOption(A, MAT_SPD, PETSC_TRUE);

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
		{
			MatSetValues(A, 1, (PetscInt *)&i, 1, (PetscInt *)&graph.csr_column[j], (PetscScalar *)&graph.csr_value[j], INSERT_VALUES);
		}
	}

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	SVDCreate(PETSC_COMM_WORLD, &svd);
	SVDSetOperator(svd, A);

	SVDSetDimensions(svd, (PetscInt)dimension, PETSC_DEFAULT, PETSC_DEFAULT);

	SVDSolve(svd);

	PetscInt nconv = 0;
	SVDGetConverged(svd, &nconv);
	PetscPrintf(PETSC_COMM_WORLD, "Number of converged solutions = %d\n\n", (int)nconv);

	// Retrieve decomposition
	// Eigenvecs are distributed amog processors and
	// will be gathered to root (0-rank) processor via scattering
	VecScatter ctx;
	Vec u, u_global;
	PetscReal sigma = 0;
	VecCreate(PETSC_COMM_WORLD, &u);
	VecSetType(u, VECMPI);
	VecSetSizes(u, PETSC_DECIDE, (PetscInt)graph.num_nodes);

	// Create scatter-to-zero context
	VecScatterCreateToZero(u, &ctx, &u_global);

	for (PetscInt j = 0; j < dimension; j++)
	{
		SVDGetSingularTriplet(svd, j, &sigma, u, NULL);
		S->val[j] = sigma;
		VecScatterBegin(ctx, u, u_global, INSERT_VALUES, SCATTER_FORWARD);
		VecScatterEnd(ctx, u, u_global, INSERT_VALUES, SCATTER_FORWARD);
/*		if ( j == 0 && rank == 2)
			for (PetscInt i = graph.num_nodes-10; i <graph.num_nodes ; i++)
			{
				double temp = 0;
				VecGetValues(u, 1, (const PetscInt *)&i, (PetscScalar *)&temp);
				printf("%lf ", temp);
			}     */
		if (rank == 0)  
		{
			for (PetscInt i = 0; i < graph.num_nodes; i++)
				VecGetValues(u_global, 1, (const PetscInt *)&i, (PetscScalar *)&U->val[j][i]);
		}
		//PetscPrintf(PETSC_COMM_WORLD, " sv %lf\n\n", (double) sigma );
	}

/*
	printf("rank %d\n\n", rank);
	PetscInt size;
	VecGetLocalSize(u, &size);
	printf("local size %d\n", (int)size);
	for (int i = graph.num_nodes-10; i <graph.num_nodes; i++)
		printf("%lf ", U->val[0][i]);
	printf("\n\n"); */

	VecScatterDestroy(&ctx);

	SVDDestroy(&svd);
	MatDestroy(&A);
	VecDestroy(&u);
	VecDestroy(&u_global);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*----------------------------------------------------------------------------------------------------------------------*/

//	GENERATING GRAPH SPLITS  //
/*---------------------------------------------------------------------------------------------------------------------*/

//struct that is passed to each AdaDIF_rec thread
typedef struct
{
	csr_graph_t graph;
	csr_graph_t *graph_out;
	edge_tuple_t **pos_edges;
	edge_tuple_t **neg_edges;
	sz_med_t num_edges;
	sz_med_t from_split;
	sz_med_t local_num_splits;
	bool test;
} split_gen_thread_type_t;

static void *generate_graph_splits_single_thread(void *);
static sz_short_t get_threads_and_width(sz_med_t *, sz_long_t);
static void edge_sampling(csr_graph_t, csr_graph_t *, edge_tuple_t *, edge_tuple_t *, sz_med_t, bool);
static sz_long_t rand_lim(sz_long_t);
static void remove_from_list_long(sz_long_t *, const sz_long_t *, const sz_long_t *, sz_long_t, sz_long_t);
static void remove_from_list_double(double *, const double *, const sz_long_t *, sz_long_t, sz_long_t);

//Function that outputs a number of graphs by randomly sampling and removing edges from input graph
// Input graph is not normalized
// Output graphs are normalized
// Edges are sampled such that resulting graphs do not contain isolated nodes
// Spawns multiple threads, each one working on a local graph copy
static csr_graph_t *generate_graph_splits(csr_graph_t graph, edge_tuple_t **pos_edges, edge_tuple_t **neg_edges, cmd_args_t args)
{

	if (args.test)
	{
		print_adj(graph);
		printf("\n%" PRIu64 "\n", graph.csr_row_pointer[graph.num_nodes]);
	}

	//unpack arguments
	sz_med_t splits = args.splits;
	sz_med_t edges_per_split = args.edges_per_split;
	bool single_thread = args.single_thread;
	sz_med_t width;
	sz_short_t num_threads = get_threads_and_width(&width, (sz_long_t)splits);

	if (single_thread)
	{
		num_threads = 1;
		width = splits;
	}

	csr_graph_t *graph_samples = (csr_graph_t *)malloc(splits * sizeof(csr_graph_t));
	for (sz_med_t i = 0; i < splits; i++)
	{
		csr_alloc(&graph_samples[i]);
		csr_realloc(&graph_samples[i], graph.nnz - 2 * edges_per_split, graph.num_nodes);
	}

	csr_graph_t *graph_copy = csr_mult_deep_copy(graph, num_threads);

	//Prepare data to be passed to each thread
	split_gen_thread_type_t parameters[num_threads];

	for (sz_short_t i = 0; i < num_threads; i++)
	{
		parameters[i] = (split_gen_thread_type_t){.graph = graph_copy[i],
												  .graph_out = graph_samples,
												  .pos_edges = pos_edges,
												  .neg_edges = neg_edges,
												  .num_edges = edges_per_split,
												  .from_split = i * width,
												  .local_num_splits = width,
												  .test = args.test};
	}
	parameters[num_threads - 1].local_num_splits = splits - (num_threads - 1) * width;

	//Spawn threads and start running
	pthread_t tid[num_threads];
	for (sz_short_t i = 0; i < num_threads; i++)
	{
		pthread_create(&tid[i], NULL, generate_graph_splits_single_thread, (void *)(parameters + i));
	}

	//Wait for all threads to finish before continuing
	for (sz_short_t i = 0; i < num_threads; i++)
	{
		pthread_join(tid[i], NULL);
	}

	//free graph copies
	csr_array_free(graph_copy, num_threads);

	return graph_samples;
}

//Each thread may generate multiple splits depending on whether num_cores<splits
static void *generate_graph_splits_single_thread(void *param)
{

	split_gen_thread_type_t *data = param;

	srand(0); //seed the random number generator

	for (sz_med_t i = 0; i < data->local_num_splits; i++)
		edge_sampling(data->graph, data->graph_out + data->from_split + i, data->pos_edges[data->from_split + i],
					  data->neg_edges[data->from_split + i], data->num_edges, data->test);

	pthread_exit(0);
}

//Function that actually samples the edges
//Ensures that there will be no isolated nodes
static void edge_sampling(csr_graph_t graph, csr_graph_t *graph_out, edge_tuple_t *pos_edges,
						  edge_tuple_t *neg_edges, sz_med_t num_edges, bool test)
{

	sz_long_t *degrees_temp = (sz_long_t *)malloc(graph.num_nodes * sizeof(sz_long_t));
	memcpy(degrees_temp, graph.degrees, graph.num_nodes * sizeof(sz_long_t));

	sz_long_t csr_edges_removed[2 * num_edges];

	//sample positive edges
	sz_long_t iter = 0;
	sz_med_t edges_sampled = 0;
	do
	{
		sz_long_t candid_node = rand_lim(graph.num_nodes - 1);
		if (degrees_temp[candid_node] > 1)
		{

			sz_long_t candid_adjacent_node = rand_lim(graph.degrees[candid_node] - 1);
			sz_long_t candid_adj_node_ind = graph.csr_column[graph.csr_row_pointer[candid_node] + candid_adjacent_node];
			sz_long_t candid_adj_node_degree = degrees_temp[candid_adj_node_ind];
			double candid_edge_val = graph.csr_value[graph.csr_row_pointer[candid_node] + candid_adjacent_node];

			if (candid_node != candid_adj_node_ind && candid_adj_node_degree > 1 && candid_edge_val > 0.0f)
			{

				//proceed with edge removal

				//First correct termporary degrees
				degrees_temp[candid_node] -= 1;
				degrees_temp[candid_adj_node_ind] -= 1;

				//Set the two edge values to 0 and store csr locations of removed edge
				graph.csr_value[graph.csr_row_pointer[candid_node] + candid_adjacent_node] = 0.0f;
				csr_edges_removed[2 * edges_sampled] = graph.csr_row_pointer[candid_node] + candid_adjacent_node;

				for (sz_long_t i = graph.csr_row_pointer[candid_adj_node_ind]; i < graph.csr_row_pointer[candid_adj_node_ind + 1]; i++)
				{
					if (graph.csr_column[i] == candid_node)
					{
						graph.csr_value[i] = 0.0f;
						csr_edges_removed[2 * edges_sampled + 1] = i;
						break;
					}
				}

				//Store sampled edge tuple
				pos_edges[edges_sampled] = (edge_tuple_t){.node_a = candid_node,
														  .node_b = candid_adj_node_ind};

				//            printf("%"PRIu64" %"PRIu64" \n",pos_edges[edges_sampled].node_a,pos_edges[edges_sampled].node_b);

				edges_sampled += 1;
			}
		}
		iter++;
	} while ((edges_sampled < num_edges) && (iter < MAX_SAMPLE));

	if (iter >= MAX_SAMPLE)
		printf("\nMaximum number of iterations reached. Try sampling fewer edges.\n\n");

	//restore graph edges
	for (sz_long_t i = 0; i < 2 * num_edges; i++)
		graph.csr_value[csr_edges_removed[i]] = 1.0f;

	//sample negative (non-existing edges)
	edges_sampled = 0;
	do
	{
		sz_long_t candid_node_a = rand_lim(graph.num_nodes - 1);
		sz_long_t candid_node_b = rand_lim(graph.num_nodes - 1);

		bool edge_exists = false;
		for (sz_long_t i = graph.csr_row_pointer[candid_node_a]; i < graph.csr_row_pointer[candid_node_a + 1]; i++)
		{
			if (graph.csr_column[i] == candid_node_b)
			{
				edge_exists = true;
				break;
			}
		}

		if (!edge_exists)
		{
			neg_edges[edges_sampled] = (edge_tuple_t){.node_a = candid_node_a,
													  .node_b = candid_node_b};
			edges_sampled += 1;
		}

	} while (edges_sampled < num_edges);

	remove_from_list_double(graph_out->csr_value, (const double *)graph.csr_value, (const sz_long_t *)csr_edges_removed, graph.nnz, 2 * num_edges);

	remove_from_list_long(graph_out->csr_column, (const sz_long_t *)graph.csr_column, (const sz_long_t *)csr_edges_removed, graph.nnz, 2 * num_edges);

	graph_out->csr_row_pointer[0] = 0;
	for (sz_long_t i = 1; i < graph.num_nodes + 1; i++)
		graph_out->csr_row_pointer[i] = graph_out->csr_row_pointer[i - 1] + degrees_temp[i - 1];

	if (test)
	{
		print_adj(*graph_out);
		printf("\n%" PRIu64 "\n", graph_out->csr_row_pointer[graph.num_nodes]);

		printf("Edges:\n");
		for (sz_long_t i = 0; i < graph.nnz; i++)
			printf("%" PRIu64 " ", graph.csr_column[i]);
		printf("\n");

		printf("Values:\n");
		for (sz_long_t i = 0; i < graph.nnz; i++)
			printf("%d ", (int)graph.csr_value[i]);
		printf("\n");

		printf("Row pointers before:\n");
		for (sz_long_t i = 0; i < graph.num_nodes + 1; i++)
			printf("%" PRIu64 " ", graph.csr_row_pointer[i]);
		printf("\n");

		printf("Indexes removed:\n");
		for (sz_long_t i = 0; i < 2 * num_edges; i++)
			printf("%" PRIu64 " ", csr_edges_removed[i]);
		printf("\n");

		printf("Edges remaining:\n");
		for (sz_long_t i = 0; i < graph.nnz - 2 * num_edges; i++)
			printf("%" PRIu64 " ", graph_out->csr_column[i]);
		printf("\n");

		printf("Values remaining:\n");
		for (sz_long_t i = 0; i < graph.nnz - 2 * num_edges; i++)
			printf("%d ", (int)graph_out->csr_value[i]);
		printf("\n");

		printf("Row pointers after:\n");
		for (sz_long_t i = 0; i < graph.num_nodes + 1; i++)
			printf("%" PRIu64 " ", graph_out->csr_row_pointer[i]);
		printf("\n");
	}

	free(degrees_temp);
}

// Computes how many threads will be used and how many classes will be allocated per thread
static sz_short_t get_threads_and_width(sz_med_t *width, sz_long_t num_users)
{
	sz_short_t num_procs = get_nprocs();
	sz_short_t num_threads;

	if (num_users <= num_procs)
	{
		num_threads = num_users;
		*width = 1;
	}
	else
	{
		num_threads = num_procs;
		*width = (sz_med_t)ceil((double)(num_users / (double)num_threads));
	}
	return num_threads;
}

//return a random number between 0 and limit inclusive.
static sz_long_t rand_lim(sz_long_t limit)
{

	int divisor = RAND_MAX / ((int)limit + 1);
	int retval;
	do
	{
		retval = rand() / divisor;
	} while (retval > limit);
	return retval;
}

//Remove items of given indexes from array (list)
// Return result in NEW  list
static void remove_from_list_long(sz_long_t *new_list, const sz_long_t *list, const sz_long_t *indexes_to_be_removed,
								  sz_long_t len, sz_long_t num_removed)
{

	sz_short_t *mask = (sz_short_t *)malloc(len * sizeof(sz_short_t));

	memset(mask, 0, len * sizeof(sz_short_t));

	for (sz_long_t i = 0; i < num_removed; i++)
	{
		mask[indexes_to_be_removed[i]] = 1;
	}

	sz_long_t k = 0;
	for (sz_long_t i = 0; i < len; i++)
	{
		if (mask[i] == 0)
		{
			new_list[k++] = list[i];
		}
	}

	free(mask);
}

//Remove items of given indexes from array (list)
// Return result in NEW  list
static void remove_from_list_double(double *new_list, const double *list, const sz_long_t *indexes_to_be_removed,
									sz_long_t len, sz_long_t num_removed)
{

	sz_short_t *mask = (sz_short_t *)malloc(len * sizeof(sz_short_t));

	memset(mask, 0, len * sizeof(sz_short_t));

	for (sz_long_t i = 0; i < num_removed; i++)
	{
		mask[indexes_to_be_removed[i]] = 1;
	}

	sz_long_t k = 0;
	for (sz_long_t i = 0; i < len; i++)
	{
		if (mask[i] == 0)
		{
			new_list[k++] = list[i];
		}
	}

	free(mask);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*----------------------------------------------------------------------------------------------------------------------*/

//	BUILDING AND FITTING EMBEDDINGS  //
/*---------------------------------------------------------------------------------------------------------------------*/

static void fit_emb_coefficients_core(csr_graph_t, d_mat_t *, d_vec_t *, double *, edge_tuple_t *, edge_tuple_t *, cmd_args_t);
static d_vec_t pool(d_mat_t);
static d_vec_t majority_vote(d_mat_t);
static d_vec_t pool(d_mat_t);

//List of possible edge fitting methods
void (*fit_edges[NUM_METHODS])(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
							   edge_tuple_t *neg_edges, double *theta, cmd_args_t args) = {logistic_fit,
																						   ls_fit,
																						   svm_fit,
																						   single_best};

//Input: graph, embedding dimensions, and coefficients theta
//Output: node embeddings
static void build_embeddings(d_mat_t embeddings, csr_graph_t graph, d_mat_t thetas, cmd_args_t args)
{

	//Pool thetas
	d_vec_t theta;
	if (args.which_fit == 3)
		theta = majority_vote(thetas);
	else
		theta = pool(thetas);

	// Scale everything to avoid issies with sklearn
	for(int k=0; k<theta.num_entries; k++ )
		theta.val[k] += (double) SCALE;

	//Normalize graph
	csr_normalize(graph);

	//Add identity to adjacency matrix
	csr_add_diagonal(graph, 1.0);

	//Scale down to bring spectral norm to 1
	csr_scale(graph, 0.5);

	//intitialize eigenpairs
	d_mat_t eigvecs = d_mat_init((sz_long_t)args.dimension, graph.num_nodes);
	d_vec_t eigvals = d_vec_init((sz_long_t)args.dimension);

	//Perform SVD on resulting graph
	my_svd(graph, &eigvecs, &eigvals, args.dimension, args.cmd);

	//Compute node_embeddings
	double S[args.dimension];
	for (sz_med_t j = 0; j < args.dimension; j++)
		S[j] = 0.0f;
	for (sz_med_t k = 0; k < args.walk_length; k++)
	{
		for (sz_med_t j = 0; j < args.dimension; j++)
			S[j] += theta.val[k] * pow(eigvals.val[j], k + 1);
	}

	for (sz_long_t i = 0; i < embeddings.num_rows; i++)
	{
		for (sz_med_t j = 0; j < args.dimension; j++)
			embeddings.val[i][j] = S[j] * eigvecs.val[j][i];
	}

	//free
	d_vec_free(theta);
	d_mat_free(eigvecs);
	d_vec_free(eigvals);
}

//Function that fits embedding coefficients to an array of graphs and edge splits
static void fit_emb_coefficients(csr_graph_t *graph, d_mat_t *thetas, edge_tuple_t **pos_edges,
								 edge_tuple_t **neg_edges, cmd_args_t args)
{
	//intitialize eigenpairs
	d_mat_t eigvecs;
	// Only root really needs to store eigvecs
	// Rest of processes just allocate a dummy
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	eigvecs = d_mat_init((sz_long_t)args.dimension, graph[0].num_nodes);

	d_vec_t eigvals = d_vec_init((sz_long_t)args.dimension);

	for (sz_med_t i = 0; i < args.splits; i++)
		fit_emb_coefficients_core(graph[i], &eigvecs, &eigvals, thetas->val[i],
								  pos_edges[i], neg_edges[i], args);

	//free eigenpairs
	d_mat_free(eigvecs);
	d_vec_free(eigvals);
}

//Function that actually fits the coefficients to a given graph and edge split
static void fit_emb_coefficients_core(csr_graph_t graph, d_mat_t *eigvecs, d_vec_t *eigvals, double *theta,
									  edge_tuple_t *pos_edges, edge_tuple_t *neg_edges, cmd_args_t args)
{

	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (args.test)
	{
		printf("\n\nEdges removed: \n");
		for (sz_long_t i = 0; i < args.edges_per_split; i++)
			printf("%" PRIu64 " %" PRIu64 "\n", pos_edges[i].node_a, pos_edges[i].node_b);
		printf("\n\n");
	}

	//Normalize graph
	csr_normalize(graph);

	//Add identity to adjacency matrix
	csr_add_diagonal(graph, 1.0);

	//Scale down to bring spectral norm to 1
	csr_scale(graph, 0.5);

	if (args.test)
		print_adj(graph);

	//Perform SVD on resulting graph
	my_svd(graph, eigvecs, eigvals, args.dimension, args.cmd);

	if (rank == 0)
		(*fit_edges[args.which_fit])(eigvecs, eigvals, pos_edges, neg_edges, theta, args);

	//Restore graph copy in case it needs to be reused for another split
	for (sz_long_t i = 0; i < graph.nnz; i++)
		graph.csr_value[i] = 1.0f;
}

//Aggregates rows of input matrix X to vector y
static d_vec_t pool(d_mat_t X)
{

	d_vec_t y = d_vec_init(X.num_cols);

	for (sz_med_t i = 0; i < X.num_cols; i++)
	{
		for (sz_med_t j = 0; j < X.num_rows; j++)
			y.val[i] += X.val[j][i];
		y.val[i] /= (double)X.num_rows;
	}

	return y;
}

// Decide best step k by majority voting
// All others are set to 0 and the best is set to SCALE
static d_vec_t majority_vote(d_mat_t X)
{

	d_vec_t y = d_vec_init(X.num_cols);

	int best_k = 0;
	double best = 0.0f;
	for (sz_med_t i = 0; i < X.num_cols; i++)
	{
		for (sz_med_t j = 0; j < X.num_rows; j++)
			y.val[i] += X.val[j][i];
		if (y.val[i] > best)
		{
			best_k = i;
			best = y.val[i];
		}
	}

	for (sz_med_t i = 0; i < X.num_cols; i++)
		y.val[i] = 0.0f;
	y.val[best_k] = 1.0;

	return y;
}

/*----------------------------------------------------------------------------------------------------------------------*/

//	 CSR_HANDLING  //
/*---------------------------------------------------------------------------------------------------------------------*/

// Make a copy of graph with edges multiplied by some scalar
static csr_graph_t csr_deep_copy_and_scale(csr_graph_t graph, double scale)
{

	csr_graph_t graph_temp;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value = (double *)malloc(graph.nnz * sizeof(double));

	graph_temp.csr_column = (sz_long_t *)malloc(graph.nnz * sizeof(sz_long_t));

	graph_temp.csr_row_pointer = (sz_long_t *)malloc((graph.num_nodes + 1) * sizeof(sz_long_t));

	graph_temp.degrees = (sz_long_t *)malloc(graph.num_nodes * sizeof(sz_long_t));

	graph_temp.num_nodes = graph.num_nodes;

	graph_temp.nnz = graph.nnz;

	//copy data

	memcpy(graph_temp.csr_row_pointer, graph.csr_row_pointer, (graph.num_nodes + 1) * sizeof(sz_long_t));

	memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes * sizeof(sz_long_t));

	memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz * sizeof(sz_long_t));

	for (sz_long_t i = 0; i < graph.nnz; i++)
	{
		graph_temp.csr_value[i] = scale * graph.csr_value[i];
	}

	return graph_temp;
}

// Make a copy of graph with edges multiplied by some scalar
static csr_graph_t csr_deep_copy(csr_graph_t graph)
{

	csr_graph_t graph_temp;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value = (double *)malloc(graph.nnz * sizeof(double));

	graph_temp.csr_column = (sz_long_t *)malloc(graph.nnz * sizeof(sz_long_t));

	graph_temp.csr_row_pointer = (sz_long_t *)malloc((graph.num_nodes + 1) * sizeof(sz_long_t));

	graph_temp.degrees = (sz_long_t *)malloc(graph.num_nodes * sizeof(sz_long_t));

	graph_temp.num_nodes = graph.num_nodes;

	graph_temp.nnz = graph.nnz;

	//copy data

	memcpy(graph_temp.csr_row_pointer, graph.csr_row_pointer, (graph.num_nodes + 1) * sizeof(sz_long_t));

	memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes * sizeof(sz_long_t));

	memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz * sizeof(sz_long_t));

	memcpy(graph_temp.csr_value, graph.csr_value, graph.nnz * sizeof(double));

	return graph_temp;
}

//Return an array with multiple copies of the input graph
static csr_graph_t *csr_mult_deep_copy(csr_graph_t graph, sz_short_t num_copies)
{
	csr_graph_t *graph_array = (csr_graph_t *)malloc(num_copies * sizeof(csr_graph_t));
	for (sz_short_t i = 0; i < num_copies; i++)
	{
		graph_array[i] = csr_deep_copy(graph);
	}
	return graph_array;
}

//Subroutine: modify csr_value to be column stochastic
//First find degrees by summing element of each row
//Then go through values and divide by corresponding degree (only works for undirected graph)
static void make_CSR_col_stoch(csr_graph_t *graph)
{
	for (sz_long_t i = 0; i < graph->nnz; i++)
	{
		graph->csr_value[i] = graph->csr_value[i] / (double)graph->degrees[graph->csr_column[i]];
	}
}

//Normalization of type D^-1/2 * A * D^-1/2 that retains symmetricity
static void csr_normalize(csr_graph_t graph)
{

	double *true_degrees = (double *)malloc(graph.num_nodes * sizeof(double));

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		true_degrees[i] = 0.0f;
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
		{
			true_degrees[i] += graph.csr_value[j];
		}
	}

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
			graph.csr_value[j] = graph.csr_value[j] / (double)sqrt(true_degrees[graph.csr_column[j]] * true_degrees[i]);
	}

	free(true_degrees);
}

//Subroutine: take x, multiply with csr matrix from right and store result in y
static void my_CSR_matvec(double *y, double *x, csr_graph_t graph)
{

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
		y[i] = 0.0;

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
			y[i] += x[graph.csr_column[j]] * graph.csr_value[j];
	}
}

//Subroutine: take X, multiply with csr matrix from right and store result in Y
static void my_CSR_matmat(double *Y, double *X, csr_graph_t graph, sz_med_t M, sz_med_t from, sz_med_t to)
{

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = from; j < to; j++)
		{
			Y[i * M + j] = 0.0f;
		}
	}

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
		{
			for (sz_med_t k = from; k < to; k++)
			{
				Y[i * M + k] += X[M * graph.csr_column[j] + k] * graph.csr_value[j];
			}
		}
	}
}

//Adds  scale*I to graph adjacency matrix
//Only works for csr graphs with preallocated diagonal
static void csr_add_diagonal(csr_graph_t graph, double scale)
{

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		bool flag = false;
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
		{
			if (graph.csr_column[j] == i)
			{
				graph.csr_value[j] = scale;
				flag = true;
				break;
			}
		}
		if (flag == false)
			printf("Warning: diagonal element not allocatedi n csr matrix\n");
	}
}

//Scales adjacency matrix

static void csr_scale(csr_graph_t graph, double scale)
{

	for (sz_long_t i = 0; i < graph.nnz; i++)
		graph.csr_value[i] *= scale;
}

/*----------------------------------------------------------------------------------------------------------------------*/

//	 TRAIN  //
/*---------------------------------------------------------------------------------------------------------------------*/

static double *build_features(d_mat_t *, d_vec_t *, edge_tuple_t *, edge_tuple_t *, cmd_args_t);
static void constr_QP_with_PG(double *, double *, double *, sz_med_t, bool);
static void grammian_matrix(double *, double *, int, int);
static void matrix_matrix_product(double *C, double *A, double *B, int, int, int);
static double cost_func(double *, double *, double *, sz_med_t);
static double max_abs_dif(double *, double *, sz_long_t);
static void project_to_simplex(double *, sz_med_t);
static void project_to_pos_quad(double *, sz_med_t);
static void matvec(double *, double *, double *, int, int);
static double frob_norm(double *, sz_med_t);
static bool detect_change(int);
static double pearson_corr(d_vec_t, d_vec_t);

//Coefficients are given by selecting the best non-zero entry of theta
//Iterative optimization here is not needed
//The performance of each k \in (1,K) can be evaluated
//and the best is selected.
static void single_best(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
						edge_tuple_t *neg_edges, double *theta, cmd_args_t args)
{

	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	for (sz_med_t i = 0; i < args.walk_length; i++)
		theta[i] = 0.0f;

	sz_med_t best_k = 0;

	double best = (double)-2 * args.edges_per_split;

	for (sz_med_t i = 0; i < args.walk_length; i++)
	{

		double metric = 0.0f;
		double eigvals_scl[args.dimension];

		for (sz_med_t j = 0; j < args.dimension; j++)
			eigvals_scl[j] = pow(eigvals->val[j], ((double)(i + 1)) / 2.0f);

		for (sz_med_t j = 0; j < args.edges_per_split; j++)
		{

			d_vec_t pos_a = d_vec_init(args.dimension);
			d_vec_t pos_b = d_vec_init(args.dimension);
			d_vec_t neg_a = d_vec_init(args.dimension);
			d_vec_t neg_b = d_vec_init(args.dimension);

			for (sz_med_t k = 0; k < args.dimension; k++)
			{
				pos_a.val[k] = eigvecs->val[k][pos_edges[j].node_a] * eigvals_scl[k];
				pos_b.val[k] = eigvecs->val[k][pos_edges[j].node_b] * eigvals_scl[k];
				neg_a.val[k] = eigvecs->val[k][neg_edges[j].node_a] * eigvals_scl[k];
				neg_b.val[k] = eigvecs->val[k][neg_edges[j].node_b] * eigvals_scl[k];
			}

			metric += pearson_corr(pos_a, pos_b);
			metric -= pearson_corr(neg_a, neg_b);

			d_vec_free(pos_a);
			d_vec_free(pos_b);
			d_vec_free(neg_a);
			d_vec_free(neg_b);
		}

		if (metric > best)
		{
			best_k = i;
			best = metric;
		}
	}

	if (rank == 0)
		printf("\n\nOptimal step for this split is: %d \n\n", (int)(best_k + 1));

	theta[best_k] = 1.0f;
}

//Compute coefficients using constrained hinge loss (SVM-like) : sum{ max(0, 1 - y * c^T * theta ) }
// from prediction of removed (pos_edges) edges and non-existing (neg_edges) ones
// Assumes first half of C corresponds to pos_edges
static void svm_fit(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
					edge_tuple_t *neg_edges, double *theta, cmd_args_t args)
{

	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	double *C = build_features(eigvecs, eigvals, pos_edges, neg_edges, args);

	sz_med_t K = args.walk_length;
	double epsilon = HINGE_EPSILON;
	double x_prev[K];
	double x[K];
	double gradient[K];
	double p[2 * args.edges_per_split];
	int total_err;

	//Initialize to uniform
	for (sz_med_t k = 0; k < K; k++)
		x[k] = 1.0f / (double)K;

	sz_med_t iter = 0;
	memcpy(x_prev, x, K * sizeof(double));
	do
	{
		iter++;

		double step_size = STEPSIZE_SVM / (double)sqrt(iter);

		//Take gradient step
		matvec(p, C, x, 2 * args.edges_per_split, K);

		total_err = 0;
		for (sz_med_t i = 0; i < args.edges_per_split; i++)
		{
			double err = epsilon - p[i];
			p[i] = (err >= 0.0f) ? 1.0 : 0.0f;
			total_err += (int)p[i];
		}
		for (sz_med_t i = args.edges_per_split; i < 2 * args.edges_per_split; i++)
		{
			double err = epsilon + p[i];
			p[i] = (err >= 0.0f) ? -1.0 : 0.0f;
			total_err += (int)-p[i];
		}

		if (rank == 0)
			printf("err: %d \n", total_err);

		for (sz_med_t k = 0; k < K; k++)
		{
			gradient[k] = 0.0;
			for (sz_med_t i = 0; i < 2 * args.edges_per_split; i++)
				gradient[k] += C[i * K + k] * p[i];
		}

		double a_1 = 1.0 - 2.0 * step_size * args.lambda;
		double a_2 = step_size / (double)2 * args.edges_per_split;

		for (sz_med_t k = 0; k < K; k++)
			x[k] = a_1 * x[k] + a_2 * gradient[k];

		//project to feasible set
		if (args.simplex)
		{
			project_to_simplex(x, K);
		}
		else
		{
			project_to_pos_quad(x, K);
		}

		memcpy(x_prev, x, K * sizeof(double));

	} while (iter < MAXIT_GD && detect_change(total_err));

	if (rank == 0)
		printf("\n Optimization finished after: %" PRIu32 " iterations\n", (uint32_t)iter);

#if PRINT_THETAS
	printf("\ntheta: ");
	print_array(x, 1, K);
	printf("\n");
#endif

	memcpy(theta, x, K * sizeof(double));

	free(C);
}

//Compute coefficients using simplex constrained logistic regression || b - C * theta  ||_2
// from prediction of removed (pos_edges) edges and non-existing (neg_edges) ones
// Assumes first half of C corresponds to pos_edges
static void logistic_fit(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
						 edge_tuple_t *neg_edges, double *theta, cmd_args_t args)
{

	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	double *C = build_features(eigvecs, eigvals, pos_edges, neg_edges, args);

	sz_med_t K = args.walk_length;
	double inf_norm, step_size;
	double x_prev[K];
	double x[K];
	double gradient[K];
	double p[2 * args.edges_per_split];

	step_size = STEPSIZE_LOG;

	//Initialize to uniform
	for (sz_med_t k = 0; k < K; k++)
		x[k] = 1.0f / (double)K;

	sz_med_t iter = 0;
	memcpy(x_prev, x, K * sizeof(double));
	do
	{
		iter++;

		//Take gradient step
		matvec(p, C, x, 2 * args.edges_per_split, K);

		for (sz_med_t i = 0; i < args.edges_per_split; i++)
			p[i] = 1.0 / (1.0 + exp(p[i]));
		for (sz_med_t i = args.edges_per_split; i < 2 * args.edges_per_split; i++)
			p[i] = -1.0 / (1.0 + exp(-p[i]));

		for (sz_med_t k = 0; k < K; k++)
		{
			gradient[k] = 0.0;
			for (sz_med_t i = 0; i < 2 * args.edges_per_split; i++)
				gradient[k] += C[i * K + k] * p[i];
		}

		double a_1 = 1.0 - 2.0 * step_size * args.lambda;
		double a_2 = step_size / (double)2 * args.edges_per_split;

		for (sz_med_t k = 0; k < K; k++)
			x[k] = a_1 * x[k] + a_2 * gradient[k];

		//project to feasible set
		if (args.simplex)
		{
			project_to_simplex(x, K);
		}
		else
		{
			project_to_pos_quad(x, K);
		}

		inf_norm = max_abs_dif(x_prev, x, (sz_long_t)K);

		memcpy(x_prev, x, K * sizeof(double));

	} while (iter < MAXIT_GD && inf_norm > GD_TOL_2);

	if (rank == 0)
		printf("\n Optimization finished after: %" PRIu32 " iterations\n", (uint32_t)iter);

#if PRINT_THETAS
	printf("\ntheta: ");
	print_array(x, 1, K);
	printf("\n");
#endif

	memcpy(theta, x, K * sizeof(double));

	free(C);
}

//Compute coefficients using simplex constrained quadratic cost || b - C * theta  ||_2
// from prediction of removed (pos_edges) edges and non-existing (neg_edges) ones
// Assumes first half of C corresponds to pos_edges
static void ls_fit(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
				   edge_tuple_t *neg_edges, double *theta, cmd_args_t args)
{
	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	double *C = build_features(eigvecs, eigvals, pos_edges, neg_edges, args);

	//Compute coefficients of quadratic cost
	double A[args.walk_length * args.walk_length];
	double b[args.walk_length];

	//A = C'*C
	grammian_matrix(A, C, (int)2 * args.edges_per_split, (int)args.walk_length);

	for (sz_med_t i = 0; i < args.walk_length; i++)
		A[i * args.walk_length + i] += args.lambda;

	//b = -2* y'*case
	for (sz_med_t i = 0; i < args.walk_length; i++)
	{
		b[i] = 0.0f;

		//Top half of array C for edges that are present
		for (sz_med_t j = 0; j < args.edges_per_split; j++)
			b[i] -= C[j * args.walk_length + i];

		//Bottom half of array C for edges that are not present
		for (sz_med_t j = args.edges_per_split; j < 2 * args.edges_per_split; j++)
			b[i] += C[j * args.walk_length + i];

		b[i] *= 2.0f;
	}

	//Solve the simplex constrained QP
	constr_QP_with_PG(theta, A, b, args.walk_length, args.simplex);

#if DEBUG
	printf("\nA: ");
	print_array(A, (int)args.walk_length, (int)args.walk_length);
	printf("\nb: ");
	print_array(b, 1, (int)args.walk_length);
#endif

#if PRINT_THETAS
	printf("\ntheta: ");
	print_array(theta, 1, (int)args.walk_length);
	printf("\n");
#endif

	free(C);
}

// For methods that require optimization
// This step simplifies the process by intriducing the intermediate C matrix
static double *build_features(d_mat_t *eigvecs, d_vec_t *eigvals, edge_tuple_t *pos_edges,
							  edge_tuple_t *neg_edges, cmd_args_t args)
{
	//Build matrix X of entrywise products
	double *X = (double *)malloc(2 * args.edges_per_split * args.dimension * sizeof(double));
	for (sz_med_t i = 0; i < args.edges_per_split; i++)
	{
		sz_long_t node_a = pos_edges[i].node_a;
		sz_long_t node_b = pos_edges[i].node_b;
		for (sz_med_t j = 0; j < args.dimension; j++)
			X[i * args.dimension + j] = eigvecs->val[j][node_a] * eigvecs->val[j][node_b];
	}

	for (sz_med_t i = 0; i < args.edges_per_split; i++)
	{
		sz_long_t node_a = neg_edges[i].node_a;
		sz_long_t node_b = neg_edges[i].node_b;
		for (sz_med_t j = 0; j < args.dimension; j++)
			X[(i + args.edges_per_split) * args.dimension + j] = eigvecs->val[j][node_a] * eigvecs->val[j][node_b];
	}

	//Build matrix S of eigenvalue powers
	double S[args.dimension * args.walk_length];
	for (sz_med_t i = 0; i < args.dimension; i++)
	{
		for (sz_med_t j = 0; j < args.walk_length; j++)
			S[i * args.walk_length + j] = pow(eigvals->val[i], j + 1);
		//      S[i*args.walk_length + j] = (i==j) ? 1.0f : 0.0;
	}

	//Intermediate matrix C = X* S
	double *C = (double *)malloc(2 * args.edges_per_split * args.walk_length * sizeof(double));

	matrix_matrix_product(C, X, S, (int)2 * args.edges_per_split,
						  (int)args.dimension, (int)args.walk_length);

	free(X);

	return C;
}

//Solving constrained quadratic minimization via projected gradient descent
//Constrains are either probability simplex (simplex = TRUE) or positive quadrant (simplex = FALSE)
//Used by AdaDIF and AdaDIF_LOO
// The following function returns x =arg min {x^T*A*x +x^T*B} s.t. x in Prob. Simplex
static void constr_QP_with_PG(double *x, double *A, double *b, sz_med_t K, bool simplex)
{
	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	double inf_norm, step_size;
	double x_temp[K];
	double x_prev[K];

	//	step_size = STEPSIZE_2;
	//	step_size = STEPSIZE_2/pow(frob_norm(A,K), 2.0f);
	step_size = STEPSIZE_2 / frob_norm(A, K);

	//Initialize to uniform
	for (sz_med_t i = 0; i < K; i++)
		x[i] = 1.0f / (double)K;

	sz_med_t iter = 0;
	memcpy(x_prev, x, K * sizeof(double));
	do
	{
		iter++;

		//Take gradient step
		matvec(x_temp, A, x, K, K);

		for (sz_med_t j = 0; j < K; j++)
			x[j] -= step_size * (2.0f * x_temp[j] + b[j]);

		//project to feasible set
		if (simplex)
		{
			project_to_simplex(x, K);
		}
		else
		{
			project_to_pos_quad(x, K);
		}

#if DEBUG
		printf("\n COST: ");
		printf(" %lf ", cost_func(A, b, x, K));
#endif

		inf_norm = max_abs_dif(x_prev, x, (sz_long_t)K);

		memcpy(x_prev, x, K * sizeof(double));

	} while (iter < MAXIT_GD && inf_norm > GD_TOL_2);

	if (rank == 0)
		printf("\n Optimization finished after: %" PRIu32 " iterations\n", (uint32_t)iter);
}

// Grammian matrix G =A'*A using CBLAS
// A : m x n
static void grammian_matrix(double *G, double *A, int m, int n)
{

	for (int i = 0; i < n * n; i++)
		G[i] = 0.0f;

	double *A_copy = (double *)malloc(m * n * sizeof(double));

	memcpy(A_copy, A, m * n * sizeof(double));

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n,
				n, m, 1.0f, A_copy, n, A, n, 0.0f, G, n);

	free(A_copy);
}

//frobenious norm of double-valued square matrix
static double frob_norm(double *A, sz_med_t dim)
{
	double norm = 0.0f;

	for (sz_med_t i = 0; i < dim * dim; i++)
	{
		norm += pow(A[i], 2.0f);
	}

	return sqrt(norm);
}

//Interface for CBLAS matrix vector product
// A : M x N
static void matvec(double *y, double *A, double *x, int M, int N)
{

	for (int i = 0; i < M; i++)
	{
		y[i] = 0.0f;
	}

	cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, x, 1, 0.0f, y, 1);
}

//Project vector onto simplex by alternating projections onto line and positive quadrant
//Operation happens in place
static void project_to_simplex(double *x, sz_med_t N)
{
	double sum, a;
	sz_short_t flag;

	do
	{
		flag = 0;
		sum = 0.0f;

		for (sz_med_t i = 0; i < N; i++)
			sum += x[i];

		a = (sum - 1.0f) / (double)N;

		for (sz_med_t i = 0; i < N; i++)
		{
			x[i] -= a;
			if (x[i] <= -PROJ_TOL)
			{
				x[i] = 0.0f;
				flag = 1;
			}
		}

	} while (flag == 1);
}

//Project vector x onto positive quadrant (truncate neg entries to 0.0)
//Operation happens in place
static void project_to_pos_quad(double *x, sz_med_t N)
{

	for (sz_med_t i = 0; i < N; i++)
		x[i] = (x[i] >= 0.0f) ? x[i] : 0.0;
}

//Evaluates quadtratic with Hessian A and linear part b at x

static double cost_func(double *A, double *b, double *x, sz_med_t len)
{

	double quad = 0.0f, lin = 0.0f;

	for (sz_med_t i = 0; i < len; i++)
	{
		for (sz_med_t j = 0; j < len; j++)
		{
			quad += A[i * len + j] * x[i] * x[j];
		}
		lin += b[i] * x[i];
	}
	return quad + lin;
}

//Infinity norm

static double max_abs_dif(double *a, double *b, sz_long_t len)
{
	double dif, max = 0.0;

	for (sz_long_t i = 0; i < len; i++)
	{
		dif = fabs(a[i] - b[i]);
		max = (dif > max) ? dif : max;
	}

	return max;
}

// Returns TRUE if there a change larger than a given threshold
// between any two consequtive inputs.
// Forgets changes after N inputs

bool detect_change(int x)
{

	bool change = true;

	static int last_input = 0;
	static int time_since_last_change = 0;

	if (abs(x - last_input) > DC_THRES)
	{
		time_since_last_change = 0;
	}
	else
	{
		time_since_last_change++;
	}

	if (time_since_last_change > DC_WIN_LEN)
		change = false;

	last_input = x;

	return change;
}

//Interface for CBLAS mutrix matrix product
// C =A*B
// A : m x k
// B : k x n
static void matrix_matrix_product(double *C, double *A, double *B, int m, int k, int n)
{

	for (int i = 0; i < m * n; i++)
		C[i] = 0.0f;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,
				n, k, 1.0f, A, k, B, n, 0.0f, C, n);
}

//Pearson correlation between two vectors
double pearson_corr(d_vec_t a, d_vec_t b)
{

	assert(a.num_entries == b.num_entries);

	double corr, dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

	int N = a.num_entries;

	for (int i = 0; i < N; i++)
	{
		dot += a.val[i] * b.val[i];
		norm_a += a.val[i] * a.val[i];
		norm_b += b.val[i] * b.val[i];
	}

	corr = dot / sqrt(norm_a * norm_b);

	return corr;
}

/*----------------------------------------------------------------------------------------------------------------------*/

//	I/O ROUTINES  //
/*---------------------------------------------------------------------------------------------------------------------*/

static sz_long_t edge_list_to_csr(sz_long_t **, double *, sz_long_t *, sz_long_t *, sz_long_t, sz_long_t *, sz_long_t *);
static sz_long_t read_adjacency_to_buffer(sz_long_t **, FILE *);
static sz_long_t **give_edge_list(char *, sz_long_t *);
static int compare(const void *, const void *);
static int file_isreg(char *);

#define CONFIG_FILE "config_input"

//List of methods ( MUST be aligned with method array in emb_fit.c )
static const char *method_list[] = {
	"logistic",
	"ls",
	"svm",
	"single_best"};

//Parsing config_input fiel to get input arguments
static void parse_config_input_file(cmd_args_t *args)
{

	// Get rank of current communicator
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	//set default arguments
	(*args) = (cmd_args_t){.walk_length = DEFAULT_NUM_WALK,
						   .graph_file = DEFAULT_GRAPH,
						   .single_thread = DEFAULT_SINGLE_THREAD,
						   .outfile = DEFAULT_OUTFILE,
						   .splits = DEFAULT_SPLITS,
						   .edges_per_split = DEFAULT_EDGES_PER_SPLIT,
						   .dimension = DEFAULT_DIMENSION,
						   .test = DEFAULT_TEST,
						   .lambda = DEFAULT_LAMBDA,
						   .simplex = DEFAULT_SIMPLEX,
						   .which_fit = DEFAULT_FIT};

	char ignore[1024];
	char stemp[50];
	int temp = 0;

	FILE *file = fopen(CONFIG_FILE, "r");

	if (file == NULL)
	{
		if (rank == 0)
			printf("ERROR: Could not find config_input file\n");
		exit(EXIT_FAILURE);
	}

	if (rank == 0)
		printf("\n\n INPUT PARAMETERS (from config_input): \n\n");

	// Graph file
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	args->graph_file = (char *)malloc(264 * sizeof(char));
	fscanf(file, "%s\n", args->graph_file);
	if (file_isreg(args->graph_file) != 1)
	{
		if (rank == 0)
			printf("ERROR: %s does not exist\n", args->graph_file);
		exit(EXIT_FAILURE);
	}
	if (rank == 0)
		printf("%s\n\n", args->graph_file);

	// Output file
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	args->outfile = (char *)malloc(264 * sizeof(char));
	fscanf(file, "%s\n", args->outfile);
	if (rank == 0)
		printf("%s\n\n", args->outfile);

	// Dimension
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	fscanf(file, "%d\n", &temp);
	args->dimension = (sz_med_t)temp;
	if (rank == 0)
		printf("%d\n\n", (int)args->dimension);

	// Walk length
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n", ignore);
	fscanf(file, "%d\n", &temp);
	args->walk_length = (sz_med_t)temp;
	if (rank == 0)
		printf(" %d\n\n", (int)args->walk_length);

	// Number of splits
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	fscanf(file, "%d\n", &temp);
	args->splits = (sz_med_t)temp;
	if (rank == 0)
		printf("%d\n\n", (int)args->splits);

	// Number of edges per split
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	fscanf(file, "%d\n", &temp);
	args->edges_per_split = (sz_med_t)temp;
	if (rank == 0)
		printf("%d\n\n", (int)args->edges_per_split);

	// Fit with..
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	bool method_found = false;
	fscanf(file, "%s\n", stemp);
	for (sz_short_t i = 0; i < NUM_METHODS; i++)
	{
		if (!strcmp(stemp, method_list[i]))
		{
			args->which_fit = i;
			method_found = true;
		}
	}
	if (!method_found)
	{
		printf("ERROR: Fit not recognized\n");
		exit(EXIT_FAILURE);
	}
	printf("%s(%d)\n\n", stemp, (int)args->which_fit);

	// Test or not?
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	fscanf(file, "%d\n", &temp);
	args->test = (bool)temp;
	if (temp)
	{
		if (rank == 0)
			printf("Yes \n\n");
	}
	else
	{
		if (rank == 0)
			printf("No \n\n");
	}

	// Lambda
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	fscanf(file, "%lf\n", &args->lambda);
	if (rank == 0)
		printf("%lf\n\n", args->lambda);

	// Simplex or not?
	fgets(ignore, sizeof(ignore), file);
	if (rank == 0)
		printf("%s\n ", ignore);
	fscanf(file, "%d\n", &temp);
	args->simplex = (bool)temp;
	if (temp)
	{
		if (rank == 0)
			printf("Yes \n\n");
	}
	else
	{
		if (rank == 0)
			printf("No \n\n");
	}

	fclose(file);
}

//Allocate memory and create csr_graph_t from edgelist input
static csr_graph_t csr_from_edgelist_file(char *filename)
{

	//Read edgelist from file get total number of edges
	sz_long_t count;
	sz_long_t **edgelist = give_edge_list(filename, &count);

	//Find maxnimum index in edgelist-> Will be usefull for allocating diagonal
	sz_long_t max_node = 0;
	for (sz_long_t i = 0; i < count; i++)
	{
		if (edgelist[i][0] > max_node)
		{
			max_node = edgelist[i][0];
		}
		else if (edgelist[i][1] > max_node)
		{
			max_node = edgelist[i][1];
		}
	}

	edgelist = (sz_long_t **)realloc(edgelist, (count + max_node) * sizeof(sz_long_t *));
	for (sz_long_t i = count; i < count + max_node; i++)
		edgelist[i] = (sz_long_t *)malloc(2 * sizeof(sz_long_t));

	//make place for diagonal
	for (sz_long_t i = 0; i < max_node; i++)
	{
		edgelist[i + count][0] = i + 1;
		edgelist[i + count][1] = i + 1;
	}

	csr_graph_t graph;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	csr_alloc(&graph);

	//Convert undirected edge list to CSR format and return graph size
	graph.num_nodes = edge_list_to_csr(edgelist, graph.csr_value, graph.csr_column, graph.csr_row_pointer,
									   count + max_node, &graph.nnz, graph.degrees);

	csr_realloc(&graph, graph.nnz, graph.num_nodes);

	//free
	for (sz_long_t i = 0; i < count + max_node; i++)
		free(edgelist[i]);
	free(edgelist);

	//Diagonal should be by default zero
	csr_add_diagonal(graph, 0.0);

	return graph;
}

// Convert directed edgelist into undirected csr_matrix
// Also allocated space for possible diagonal elements to be added
static sz_long_t edge_list_to_csr(sz_long_t **edge, double *csr_value, sz_long_t *csr_column,
								  sz_long_t *csr_row_pointer, sz_long_t len, sz_long_t *nnz, sz_long_t *degrees)
{

	//Start bu making a 2D array twice the size where (i,j) exists for every (j,i)
	sz_long_t count_nnz;
	sz_long_t **edge_temp = (sz_long_t **)malloc(2 * len * sizeof(sz_long_t *));
	for (sz_long_t i = 0; i < 2 * len; i++)
		edge_temp[i] = (sz_long_t *)malloc(2 * sizeof(sz_long_t));

	//Mirror directed edges
	for (sz_long_t i = 0; i < len; i++)
	{
		edge_temp[i][0] = edge[i][0];
		edge_temp[i][1] = edge[i][1];
		edge_temp[i + len][1] = edge[i][0];
		edge_temp[i + len][0] = edge[i][1];
	}

	//QuickSort buffer_temp with respect to first column (Study and use COMPARATOR function for this)
	qsort(edge_temp, 2 * len, sizeof(edge_temp[0]), compare);

	//The first collumn of sorted array readily gives csr_row_pointer (just loop through and look for j s.t. x[j]!=x[j-1])
	//Not sure yet but i probably need to define small dynamic subarray with elements of second collumn and
	// sort it before stacking it it into csr_column (A: I dont need to)
	//This can all be done together in one loop over sorted buffer_temp
	csr_row_pointer[0] = 0;
	csr_value[0] = 1.0;
	csr_column[0] = edge_temp[0][1] - 1;
	sz_long_t j = 1;
	count_nnz = 1;
	for (sz_long_t i = 1; i < 2 * len; i++)
	{
		if (!(edge_temp[i - 1][0] == edge_temp[i][0] && edge_temp[i - 1][1] == edge_temp[i][1]))
		{
			csr_value[count_nnz] = 1.0;
			csr_column[count_nnz] = edge_temp[i][1] - 1;
			if (edge_temp[i][0] != edge_temp[i - 1][0])
			{
				csr_row_pointer[j] = count_nnz;
				j++;
			}
			count_nnz++;
		}
	}
	csr_row_pointer[j] = count_nnz;
	*nnz = count_nnz;

	for (sz_long_t i = 0; i < j; i++)
	{
		degrees[i] = csr_row_pointer[i + 1] - csr_row_pointer[i];
	}

	//Free temporary list
	for (sz_long_t i = 0; i < 2 * len; i++)
	{
		free(edge_temp[i]);
	}
	free(edge_temp);
	return j;
}

//Return edge list and count
static sz_long_t **give_edge_list(char *filename, sz_long_t *count)
{
	//Get process rank
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	sz_long_t **buffer = (sz_long_t **)malloc(EDGE_BUFF_SIZE * sizeof(sz_long_t *));

	for (sz_long_t i = 0; i < EDGE_BUFF_SIZE; i++)
		buffer[i] = (sz_long_t *)malloc(2 * sizeof(sz_long_t));

	FILE *file = fopen(filename, "r");

	assert(file != NULL);

	// Read adjacency into buffer into buffer and return length count=edges
	*count = read_adjacency_to_buffer(buffer, file);
	if (rank == 0)
		printf("Number of edges: %" PRIu64 "\n\n", (uint64_t)*count);

	//print_edge_list( buffer, *count);

	//Free excess memory
	for (sz_long_t i = *count + 1; i < EDGE_BUFF_SIZE; i++)
	{
		free(buffer[i]);
	}
	buffer = realloc(buffer, (*count) * sizeof(sz_long_t *));

	return buffer;
}

//Read .txt file into buffer
static sz_long_t read_adjacency_to_buffer(sz_long_t **buffer, FILE *file)
{
	sz_long_t count = 0;
	for (; count < EDGE_BUFF_SIZE; ++count)
	{
		int got = fscanf(file, "%" SCNu64 "\t%" SCNu64 "\n", &buffer[count][0], &buffer[count][1]);
		if ((got != 2) || ((buffer[count][0] == 0) && (buffer[count][1] == 0)))
			break;
		// Stop scanning if wrong number of tokens (maybe end of file) or zero input
	}
	fclose(file);
	return count;
}

//Check if file is valid
static int file_isreg(char *path)
{
	struct stat st;

	if (stat(path, &st) < 0)
		return -1;

	return S_ISREG(st.st_mode);
}

//Writes double matrix to file
static void write_d_mat(d_mat_t mat, char *filename)
{
	FILE *file = fopen(filename, "w");
	assert(file != NULL);
	for (sz_long_t i = 0; i < mat.num_rows; i++)
	{
		for (sz_long_t j = 0; j < mat.num_cols; j++)
			fprintf(file, "%lf ", mat.val[i][j]);
		fprintf(file, "\n");
	}
	fclose(file);
}

//My comparator for two collumn array. Sorts second col according to first
static int compare(const void *pa, const void *pb)
{
	const sz_long_t *a = *(const sz_long_t **)pa;
	const sz_long_t *b = *(const sz_long_t **)pb;
	if (a[0] == b[0])
		return a[1] - b[1];
	else
		return a[0] - b[0];
}

//print adjacency matrix on screen
static void print_adj(csr_graph_t graph)
{

	printf("\nAdjacency matrix:\n\n");

	bool integer = true;
	for (sz_long_t i = 0; i < graph.nnz; i++)
	{
		if (graph.csr_value[i] > 0.0 && graph.csr_value[i] < 1.0)
			integer = false;
	}

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		double row[graph.num_nodes];
		for (sz_long_t j = 0; j < graph.num_nodes; j++)
			row[j] = 0.0f;

		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
			row[graph.csr_column[j]] = graph.csr_value[j];

		for (sz_long_t j = 0; j < graph.num_nodes; j++)
		{
			if (integer)
			{
				printf("%d ", (int)row[j]);
			}
			else
			{
				printf("%lf ", row[j]);
			}
		}

		printf("\n");
	}
}

//print array of double Values
static void print_array(double *A, int rows, int cols)
{

	printf(" %d x %d array: \n", rows, cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			printf(" %lf ", A[i * cols + j]);
		printf("\n");
	}
}

/*----------------------------------------------------------------------------------------------------------------------*/

//	MEMORY MANAGEMENT //
/*---------------------------------------------------------------------------------------------------------------------*/

//Alocate memory for csr
static void csr_alloc(csr_graph_t *graph)
{
	sz_long_t nnz_buff = EDGE_BUFF_SIZE, num_nodes_buff = NODE_BUFF_SIZE;

	graph->csr_value = (double *)malloc(nnz_buff * sizeof(double));
	graph->csr_column = (sz_long_t *)malloc(nnz_buff * sizeof(sz_long_t));
	graph->csr_row_pointer = (sz_long_t *)malloc(num_nodes_buff * sizeof(sz_long_t));
	graph->degrees = (sz_long_t *)malloc(num_nodes_buff * sizeof(sz_long_t));
}

//Reallocate csr memory after size is known
static void csr_realloc(csr_graph_t *graph, sz_long_t nnz, sz_long_t num_nodes)
{
	graph->num_nodes = num_nodes;
	graph->nnz = nnz;
	graph->csr_value = realloc(graph->csr_value, nnz * sizeof(double));
	graph->csr_column = realloc(graph->csr_column, nnz * sizeof(sz_long_t));
	graph->csr_row_pointer = realloc(graph->csr_row_pointer, (num_nodes + 1) * sizeof(sz_long_t));
	graph->degrees = realloc(graph->degrees, num_nodes * sizeof(sz_long_t));
}

// Free memory allocated to csr_graph_t
static void csr_free(csr_graph_t graph)
{
	free(graph.csr_value);
	free(graph.csr_column);
	free(graph.csr_row_pointer);
	free(graph.degrees);
}

// Free memory allocated to array of csr_graph_ts
static void csr_array_free(csr_graph_t *graph_array, sz_short_t num_copies)
{
	for (sz_short_t i = 0; i < num_copies; i++)
		csr_free(graph_array[i]);
	free(graph_array);
}

// Allocate double matrix
static d_mat_t d_mat_init(sz_long_t num_rows, sz_long_t num_cols)
{
	d_mat_t mat = {.num_rows = num_rows, .num_cols = num_cols};
	mat.val = (double **)malloc(num_rows * sizeof(double *));
	for (sz_long_t i = 0; i < num_rows; i++)
		mat.val[i] = (double *)malloc(num_cols * sizeof(double));

	for (sz_long_t i = 0; i < num_rows; i++)
	{
		for (sz_long_t j = 0; j < num_cols; j++)
			mat.val[i][j] = 0.0;
	}

	return mat;
}

//Free double rating_matrix_file
static void d_mat_free(d_mat_t mat)
{
	for (sz_long_t i = 0; i < mat.num_rows; i++)
		free(mat.val[i]);
	free(mat.val);
}

// Allocate double vector
static d_vec_t d_vec_init(sz_long_t num_entries)
{
	d_vec_t vec = {.num_entries = num_entries};
	vec.val = (double *)malloc(num_entries * sizeof(double));
	for (sz_long_t i = 0; i < num_entries; i++)
		vec.val[i] = 0.0;
	return vec;
}

//Free double vector
static void d_vec_free(d_vec_t vec)
{
	free(vec.val);
}

//Allocate matrix of edge_tuples
static edge_tuple_t **edge_tuple_alloc(sz_med_t num_rows, sz_med_t num_cols)
{
	edge_tuple_t **edges = (edge_tuple_t **)malloc(num_rows * sizeof(edge_tuple_t *));
	for (sz_med_t i = 0; i < num_rows; i++)
		edges[i] = (edge_tuple_t *)malloc(num_cols * sizeof(edge_tuple_t));
	return edges;
}

//free matrix of edge_tuples
static void edge_tuple_free(edge_tuple_t **edges, sz_med_t num_rows)
{
	for (sz_med_t i = 0; i < num_rows; i++)
		free(edges[i]);
	free(edges);
}

//Allocate matrix of sz_long_t type
static sz_long_t **long_mat_alloc(sz_med_t num_rows, sz_med_t num_cols)
{
	sz_long_t **long_mat = (sz_long_t **)malloc(num_rows * sizeof(sz_long_t *));
	for (sz_med_t i = 0; i < num_rows; i++)
		long_mat[i] = (sz_long_t *)malloc(num_cols * sizeof(sz_long_t));
	return long_mat;
}

//free matrix of sz_long_t type
static void long_mat_free(sz_long_t **long_mat, sz_med_t num_rows)
{
	for (sz_med_t i = 0; i < num_rows; i++)
		free(long_mat[i]);
	free(long_mat);
}
