#ifndef REC_DEFS_H_
#define REC_DEFS_H_

#include <stdbool.h>

#define DEBUG false
#define PRINT_THETAS true

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

#define DEFAULT_SINGLE_THREAD false

#define DEFAULT_TEST false

#define DEFAULT_SIMPLEX true

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


//INTEGER TYPES

typedef uint64_t sz_long; //Long range unsinged integer. Used for node and edge indexing.

typedef uint32_t sz_med; //Medium range unsinged integer. Used for random walk length,
                         //iteration indexing and seed indexing

typedef uint8_t sz_short; //Short range unsigned integer. Used for class and thread indexing.

typedef int8_t class_t; // Short integer for actual label values.



//DATA STRUCTURES

// Record of user-item matrix as an array of user STRUCTURES

typedef struct{
	sz_long node_a;
	sz_long node_b;
} edge_tuple;


//Double and index struct for sorting and keeping indexes

typedef struct{
	double val;
	int ind;
} val_and_ind;

//struct forcommand line arguments

typedef struct{
	char* graph_file;
	char* outfile;
	sz_med walk_length;
	sz_med splits;
	sz_med edges_per_split;
	sz_med dimension;
	double lambda;
	bool single_thread;
	bool test;
	bool simplex;
	sz_short which_fit;
} cmd_args;


//Csr graph struct
typedef struct{
	double* csr_value;
	sz_long* csr_column;
	sz_long* csr_row_pointer;
	sz_long  num_nodes;
	sz_long  nnz;
	sz_long* degrees;
} csr_graph;

//Double matrix
typedef struct{
	double** val;
	sz_long num_rows;
	sz_long num_cols;
} d_mat;

//Double vector
typedef struct{
	double* val;
	sz_long num_entries;
} d_vec;


#endif
