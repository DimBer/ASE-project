///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains routine to generate graph splits

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
#include <pthread.h>
#include <sys/sysinfo.h>

#include "emb_defs.h"
#include "emb_mem.h"
#include "csr_handling.h"
#include "split_gen.h"
#include "emb_IO.h"


static void* generate_graph_splits_single_thread(void* );
static sz_short get_threads_and_width(sz_med*  ,sz_long );
static void edge_sampling(csr_graph, csr_graph*, edge_tuple*, edge_tuple*, sz_med, bool);
static sz_long rand_lim(sz_long );
static void remove_from_list_long(sz_long*, const sz_long* , const sz_long* ,sz_long, sz_long );
static void remove_from_list_double(double*, const double* , const sz_long* ,sz_long, sz_long );

//Function that outputs a number of graphs by randomly sampling and removing edges from input graph
// Input graph is not normalized
// Output graphs are normalized
// Edges are sampled such that resulting graphs do not contain isolated nodes
// Spawns multiple threads, each one working on a local graph copy
csr_graph* generate_graph_splits(csr_graph graph, edge_tuple** pos_edges, edge_tuple** neg_edges, cmd_args args)
{

			if(args.test){
				print_adj(graph);
				printf("\n%"PRIu64"\n",graph.csr_row_pointer[graph.num_nodes] );
			}

      //unpack arguments
      sz_med splits = args.splits;
      sz_med edges_per_split = args.edges_per_split;
      bool single_thread = args.single_thread;
      sz_med width;
      sz_short num_threads = get_threads_and_width(&width , (sz_long) splits);

      if(single_thread){
        num_threads = 1;
        width = splits;
      }

      csr_graph* graph_samples = (csr_graph*)malloc(splits*sizeof(csr_graph));
      for(sz_med i = 0;i<splits;i++){
        csr_alloc(&graph_samples[i]);
        csr_realloc(&graph_samples[i], graph.nnz - 2*edges_per_split ,graph.num_nodes);
      }

      csr_graph* graph_copy = csr_mult_deep_copy(graph, num_threads);


      //Prepare data to be passed to each thread
      split_gen_thread_type parameters[num_threads];

      for(sz_short i=0;i<num_threads;i++){
        parameters[i]= (split_gen_thread_type) {.graph = graph_copy[i],
                                                .graph_out = graph_samples,
                                                .pos_edges = pos_edges,
                                                .neg_edges = neg_edges,
                                                .num_edges = edges_per_split,
                                                .from_split = i*width,
                                                .local_num_splits = width,
																							 	.test = args.test};
      }
      parameters[num_threads-1].local_num_splits = splits - (num_threads-1)*width;

      //Spawn threads and start running
      pthread_t tid[num_threads];
      for(sz_short i=0;i<num_threads;i++){
        pthread_create(&tid[i],NULL,generate_graph_splits_single_thread,(void*)(parameters+i));
      }

      //Wait for all threads to finish before continuing
      for(sz_short i=0;i<num_threads;i++){pthread_join(tid[i], NULL);}


      //free graph copies
      csr_array_free(graph_copy, num_threads);

      return graph_samples;
}


//Each thread may generate multiple splits depending on whether num_cores<splits
static void* generate_graph_splits_single_thread( void* param){

	split_gen_thread_type* data = param;

  srand(time(NULL)); //seed the random number generator

  for(sz_med i=0; i<data->local_num_splits;i++)
    edge_sampling(data->graph, data->graph_out + data->from_split + i, data->pos_edges[data->from_split + i],
                  data->neg_edges[data->from_split + i], data->num_edges, data->test);


	pthread_exit(0);

}

//Function that actually samples the edges
//Ensures that there will be no isolated nodes
static void edge_sampling(csr_graph graph, csr_graph* graph_out, edge_tuple* pos_edges,
                          edge_tuple* neg_edges, sz_med num_edges, bool test){

  sz_long* degrees_temp = (sz_long*) malloc(graph.num_nodes*sizeof(sz_long));
  memcpy(degrees_temp, graph.degrees, graph.num_nodes*sizeof(sz_long));

  sz_long csr_edges_removed[2*num_edges];

  //sample positive edges
  sz_long iter = 0;
  sz_med edges_sampled = 0;
  do{
      sz_long candid_node = rand_lim(graph.num_nodes-1);
      if(degrees_temp[candid_node]>1){

        sz_long candid_adjacent_node = rand_lim(graph.degrees[candid_node]-1);
        sz_long candid_adj_node_ind = graph.csr_column[ graph.csr_row_pointer[candid_node] + candid_adjacent_node];
        sz_long candid_adj_node_degree = degrees_temp[candid_adj_node_ind];
        double  candid_edge_val = graph.csr_value[ graph.csr_row_pointer[candid_node] + candid_adjacent_node ];

        if( candid_node != candid_adj_node_ind && candid_adj_node_degree > 1  && candid_edge_val > 0.0f ){

            //proceed with edge removal

            //First correct termporary degrees
            degrees_temp[candid_node]-=1;
            degrees_temp[candid_adj_node_ind]-=1;

            //Set the two edge values to 0 and store csr locations of removed edge
            graph.csr_value[ graph.csr_row_pointer[candid_node] + candid_adjacent_node ] = 0.0f;
            csr_edges_removed[2*edges_sampled] = graph.csr_row_pointer[candid_node] + candid_adjacent_node;

            for( sz_long i = graph.csr_row_pointer[candid_adj_node_ind]; i<graph.csr_row_pointer[candid_adj_node_ind+1]; i++ ){
                if(graph.csr_column[i] == candid_node){
                    graph.csr_value[i] = 0.0f;
                    csr_edges_removed[2*edges_sampled + 1] = i;
                    break;
                }
            }

            //Store sampled edge tuple
            pos_edges[edges_sampled] = (edge_tuple) { .node_a =  candid_node,
                                                      .node_b = candid_adj_node_ind };

//            printf("%"PRIu64" %"PRIu64" \n",pos_edges[edges_sampled].node_a,pos_edges[edges_sampled].node_b);

            edges_sampled +=1;
        }

      }
        iter++;
  }while( ( edges_sampled < num_edges ) && (iter < MAX_SAMPLE ) );

  if(iter >= MAX_SAMPLE ) printf("\nMaximum number of iterations reached. Try sampling fewer edges.\n\n");

  //restore graph edges
  for(sz_long i=0;i<2*num_edges;i++) graph.csr_value[csr_edges_removed[i]] = 1.0f;

  //sample negative (non-existing edges)
  edges_sampled = 0;
  do{
      sz_long candid_node_a = rand_lim(graph.num_nodes-1);
      sz_long candid_node_b = rand_lim(graph.num_nodes-1);

      bool edge_exists = false;
      for(sz_long i = graph.csr_row_pointer[candid_node_a]; i<graph.csr_row_pointer[candid_node_a+1]; i++){
        if(graph.csr_column[i] == candid_node_b){
          edge_exists = true;
          break;
        }
      }

      if(!edge_exists){
        neg_edges[edges_sampled] = (edge_tuple) { .node_a =  candid_node_a,
                                                  .node_b = candid_node_b };
        edges_sampled+=1;
      }

  }while(edges_sampled < num_edges);

	remove_from_list_double(graph_out->csr_value , (const double*) graph.csr_value, (const sz_long*) csr_edges_removed, graph.nnz, 2*num_edges);

  remove_from_list_long(graph_out->csr_column , (const sz_long*) graph.csr_column, (const sz_long*) csr_edges_removed, graph.nnz, 2*num_edges);

	graph_out->csr_row_pointer[0] = 0;
	for(sz_long i=1; i<graph.num_nodes+1;i++) graph_out->csr_row_pointer[i] = graph_out->csr_row_pointer[i-1] + degrees_temp[i-1] ;

	if(test){
		print_adj(*graph_out);
		printf("\n%"PRIu64"\n",graph_out->csr_row_pointer[graph.num_nodes] );

		printf("Edges:\n");
		for(sz_long i=0; i<graph.nnz; i++) printf("%"PRIu64" ",graph.csr_column[i]);
		printf("\n");

		printf("Values:\n");
		for(sz_long i=0; i<graph.nnz; i++) printf("%d ",(int) graph.csr_value[i]);
		printf("\n");

		printf("Row pointers before:\n");
		for(sz_long i=0; i<graph.num_nodes+1; i++) printf("%"PRIu64" ",graph.csr_row_pointer[i]);
		printf("\n");

		printf("Indexes removed:\n");
		for(sz_long i=0; i<2*num_edges; i++) printf("%"PRIu64" ",csr_edges_removed[i]);
		printf("\n");

		printf("Edges remaining:\n");
		for(sz_long i=0; i<graph.nnz - 2*num_edges; i++) printf("%"PRIu64" ",graph_out->csr_column[i]);
		printf("\n");

		printf("Values remaining:\n");
		for(sz_long i=0; i<graph.nnz- 2*num_edges; i++) printf("%d ",(int) graph_out->csr_value[i]);
		printf("\n");

		printf("Row pointers after:\n");
		for(sz_long i=0; i<graph.num_nodes+1; i++) printf("%"PRIu64" ",graph_out->csr_row_pointer[i]);
		printf("\n");
	}

  free(degrees_temp);
}


// Computes how many threads will be used and how many classes will be allocated per thread
static sz_short get_threads_and_width(sz_med* width ,sz_long num_users){
	sz_short num_procs=get_nprocs();
	sz_short num_threads;

	if(num_users<=num_procs){
		num_threads = num_users;
		*width=1;
	}else{
		num_threads = num_procs;
		*width = (sz_med)ceil((double)(num_users/(double)num_threads));
	}
	return num_threads;
}


//return a random number between 0 and limit inclusive.
static sz_long rand_lim(sz_long limit){

	int divisor = RAND_MAX/((int)limit+1);
	int retval;
	do {
		retval = rand() / divisor;
	} while (retval > limit);
	return retval;
}


//Remove items of given indexes from array (list)
// Return result in NEW  list
static void remove_from_list_long(sz_long* new_list, const sz_long* list, const sz_long* indexes_to_be_removed,
			   sz_long len, sz_long num_removed ){

	int mask[len];

	memset(mask, 0, len*sizeof(int));

	for(sz_long i =0; i<num_removed; i++){ mask[indexes_to_be_removed[i]] =1 ;}

	sz_long k=0;
	for(sz_long i =0; i<len; i++){
		if(mask[i]==0){
		    new_list[k++] = list[i];
		}
	}

}

//Remove items of given indexes from array (list)
// Return result in NEW  list
static void remove_from_list_double(double* new_list, const double* list, const sz_long* indexes_to_be_removed,
			   sz_long len, sz_long num_removed ){

	int mask[len];

	memset(mask, 0, len*sizeof(int));

	for(sz_long i =0; i<num_removed; i++){ mask[indexes_to_be_removed[i]] =1 ;}

	sz_long k=0;
	for(sz_long i =0; i<len; i++){
		if(mask[i]==0){
		    new_list[k++] = list[i];
		}
	}

}
