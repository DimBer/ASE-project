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
#include <sys/sysinfo.h>
#include <stdbool.h>
#include <assert.h>

#include "emb_IO.h"
#include "emb_defs.h"
#include "csr_handling.h"
#include "emb_mem.h"
#include "train.h"

static sz_long edge_list_to_csr(sz_long **, double *, sz_long *, sz_long *, sz_long, sz_long *, sz_long *);
static sz_long read_adjacency_to_buffer(sz_long **, FILE *);
static sz_long **give_edge_list(char *, sz_long *);
static int compare(const void *, const void *);
static int file_isreg(char *);

//List of methods ( MUST be aligned with method array in emb_fit.c )
static const char *method_list[] = {
	"logistic",
	"ls",
	"svm",
	"single_best"};

//Parsing command line arguments with getopt_long_only
void parse_commandline_args(int argc, char **argv, cmd_args *args)
{

	//set default arguments
	(*args) = (cmd_args){.walk_length = DEFAULT_NUM_WALK,
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

	int opt = 0;
	//Specifying the expected options
	//The two options l and b expect numbers as argument
	static struct option long_options[] = {
		{"graph_file", required_argument, 0, 'a'},
		{"single_thread", no_argument, 0, 'b'},
		{"outfile", required_argument, 0, 'c'},
		{"dimension", required_argument, 0, 'd'},
		{"walk_length", required_argument, 0, 'e'},
		{"splits", required_argument, 0, 'f'},
		{"edges_per_split", required_argument, 0, 'g'},
		{"test", no_argument, 0, 'h'},
		{"lambda", required_argument, 0, 'i'},
		{"simplex", no_argument, 0, 'j'},
		{"fit_with", required_argument, 0, 'k'},
		{0, 0, 0, 0}};

	int long_index = 0;
	bool method_found = false;

	while ((opt = getopt_long_only(argc, argv, "",
								   long_options, &long_index)) != -1)
	{
		switch (opt)
		{
		case 'a':
			args->graph_file = optarg;
			if (file_isreg(args->graph_file) != 1)
			{
				printf("ERROR: %s does not exist\n", args->graph_file);
				exit(EXIT_FAILURE);
			}
			break;
		case 'e':
			args->walk_length = atoi(optarg);
			if (args->walk_length < 1)
			{
				printf("ERROR: Length of walks must be >=1\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'f':
			args->splits = atoi(optarg);
			if (args->splits < 1)
			{
				printf("ERROR: # of splits must be >=1\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'g':
			args->edges_per_split = atoi(optarg);
			if (args->edges_per_split < 1)
			{
				printf("ERROR: # of edges per split must be >=1\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'd':
			args->dimension = atoi(optarg);
			if (args->dimension < 1)
			{
				printf("ERROR: Embedding dimension must be >=1\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'b':
			args->single_thread = true;
			break;
		case 'c':
			args->outfile = optarg;
			break;
		case 'h':
			args->test = true;
			break;
		case 'i':
			args->lambda = atof(optarg);
			if (args->lambda < 0.0)
			{
				printf("ERROR: Lambda must be  >= 0.0\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'j':
			args->simplex = true;
			break;
		case 'k':
			for (sz_short i = 0; i < NUM_METHODS; i++)
			{
				if (!strcmp(optarg, method_list[i]))
					args->which_fit = i;
				method_found = true;
			}
			if (!method_found)
				printf("ERROR: Edge fit method not recognized\n");
			break;
			exit(EXIT_FAILURE);
		}
	}
}

//Allocate memory and create csr_graph from edgelist input
csr_graph csr_from_edgelist_file(char *filename)
{

	//Read edgelist from file get total number of edges
	sz_long count;
	sz_long **edgelist = give_edge_list(filename, &count);

	//Find maxnimum index in edgelist-> Will be usefull for allocating diagonal
	sz_long max_node = 0;
	for (sz_long i = 0; i < count; i++)
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

	edgelist = (sz_long **)realloc(edgelist, (count + max_node) * sizeof(sz_long *));
	for (sz_long i = count; i < count + max_node; i++)
		edgelist[i] = (sz_long *)malloc(2 * sizeof(sz_long));

	//make place for diagonal
	for (sz_long i = 0; i < max_node; i++)
	{
		edgelist[i + count][0] = i + 1;
		edgelist[i + count][1] = i + 1;
	}

	csr_graph graph;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	csr_alloc(&graph);

	//Convert undirected edge list to CSR format and return graph size
	graph.num_nodes = edge_list_to_csr(edgelist, graph.csr_value, graph.csr_column, graph.csr_row_pointer,
									   count + max_node, &graph.nnz, graph.degrees);

	csr_realloc(&graph, graph.nnz, graph.num_nodes);

	//free
	for (sz_long i = 0; i < count + max_node; i++)
		free(edgelist[i]);
	free(edgelist);

	//Diagonal should be by default zero
	csr_add_diagonal(graph, 0.0);

	return graph;
}

// Convert directed edgelist into undirected csr_matrix
// Also allocated space for possible diagonal elements to be added
static sz_long edge_list_to_csr(sz_long **edge, double *csr_value, sz_long *csr_column,
								sz_long *csr_row_pointer, sz_long len, sz_long *nnz, sz_long *degrees)
{

	//Start bu making a 2D array twice the size where (i,j) exists for every (j,i)
	sz_long count_nnz;
	sz_long **edge_temp = (sz_long **)malloc(2 * len * sizeof(sz_long *));
	for (sz_long i = 0; i < 2 * len; i++)
		edge_temp[i] = (sz_long *)malloc(2 * sizeof(sz_long));

	//Mirror directed edges
	for (sz_long i = 0; i < len; i++)
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
	sz_long j = 1;
	count_nnz = 1;
	for (sz_long i = 1; i < 2 * len; i++)
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

	for (sz_long i = 0; i < j; i++)
	{
		degrees[i] = csr_row_pointer[i + 1] - csr_row_pointer[i];
	}

	//Free temporary list
	for (sz_long i = 0; i < 2 * len; i++)
	{
		free(edge_temp[i]);
	}
	free(edge_temp);
	return j;
}

//Return edge list and count
static sz_long **give_edge_list(char *filename, sz_long *count)
{

	sz_long **buffer = (sz_long **)malloc(EDGE_BUFF_SIZE * sizeof(sz_long *));

	for (sz_long i = 0; i < EDGE_BUFF_SIZE; i++)
		buffer[i] = (sz_long *)malloc(2 * sizeof(sz_long));

	FILE *file = fopen(filename, "r");

	assert(file != NULL);

	// Read adjacency into buffer into buffer and return length count=edges
	*count = read_adjacency_to_buffer(buffer, file);
	printf("Number of edges: %" PRIu64 "\n\n", (uint64_t)*count);

	//print_edge_list( buffer, *count);

	//Free excess memory
	for (sz_long i = *count + 1; i < EDGE_BUFF_SIZE; i++)
	{
		free(buffer[i]);
	}
	buffer = realloc(buffer, (*count) * sizeof(sz_long *));

	return buffer;
}

//Read .txt file into buffer
static sz_long read_adjacency_to_buffer(sz_long **buffer, FILE *file)
{
	sz_long count = 0;
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
void write_d_mat(d_mat mat, char *filename)
{
	FILE *file = fopen(filename, "w");
	assert(file != NULL);
	for (sz_long i = 0; i < mat.num_rows; i++)
	{
		for (sz_long j = 0; j < mat.num_cols; j++)
			fprintf(file, "%lf ", mat.val[i][j]);
		fprintf(file, "\n");
	}
	fclose(file);
}

//My comparator for two collumn array. Sorts second col according to first
static int compare(const void *pa, const void *pb)
{
	const sz_long *a = *(const sz_long **)pa;
	const sz_long *b = *(const sz_long **)pb;
	if (a[0] == b[0])
		return a[1] - b[1];
	else
		return a[0] - b[0];
}

//print adjacency matrix on screen
void print_adj(csr_graph graph)
{

	printf("\nAdjacency matrix:\n\n");

	bool integer = true;
	for (sz_long i = 0; i < graph.nnz; i++)
	{
		if (graph.csr_value[i] > 0.0 && graph.csr_value[i] < 1.0)
			integer = false;
	}

	for (sz_long i = 0; i < graph.num_nodes; i++)
	{
		double row[graph.num_nodes];
		for (sz_long j = 0; j < graph.num_nodes; j++)
			row[j] = 0.0f;

		for (sz_long j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
			row[graph.csr_column[j]] = graph.csr_value[j];

		for (sz_long j = 0; j < graph.num_nodes; j++)
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
void print_array(double *A, int rows, int cols)
{

	printf(" %d x %d array: \n", rows, cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			printf(" %lf ", A[i * cols + j]);
		printf("\n");
	}
}
