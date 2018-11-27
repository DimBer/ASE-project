///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains methods for fitting parameters to missing edges: 1) LS fit, 2) SVM fit, and 3) logistic regression
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
#include <assert.h>

#include "emb_defs.h"
#include "emb_IO.h"
#include "emb_mem.h"
#include "train.h"

static double* build_features(d_mat* , d_vec* , edge_tuple* , edge_tuple* , cmd_args);
static void constr_QP_with_PG(double* , double* , double* , sz_med, bool );
static void grammian_matrix(double* , double* , int , int );
static void matrix_matrix_product(double* C, double* A, double* B, int , int , int );
static double cost_func(double* , double* , double* , sz_med );
static double max_abs_dif(double*, double* , sz_long );
static void project_to_simplex( double* , sz_med );
static void project_to_pos_quad(double*  , sz_med );
static void matvec(double*, double* , double* , int , int );
static double frob_norm(double* , sz_med );
static bool detect_change(int );
static double pearson_corr(d_vec, d_vec);

//Coefficients are given by selecting the best non-zero entry of theta
//Iterative optimization here is not needed
//The performance of each k \in (1,K) can be evaluated 
//and the best is selected. 
void single_best( d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args ){

  for(sz_med i=0; i<args.walk_length; i++) theta[i] = 0.0f;

  sz_med best_k = 0;

  double best = (double) -2*args.edges_per_split;

  for(sz_med i=0; i<args.walk_length; i++){

    double metric = 0.0f;
    double eigvals_scl[args.dimension];

    for(sz_med j=0; j<args.dimension; j++) 
      eigvals_scl[j] = pow(eigvals->val[j], ((double) (i+1))/2.0f );

    for(sz_med j=0; j<args.edges_per_split; j++){

      d_vec pos_a = d_vec_init( args.dimension );
      d_vec pos_b = d_vec_init( args.dimension );
      d_vec neg_a = d_vec_init( args.dimension );
      d_vec neg_b = d_vec_init( args.dimension );

      for(sz_med k=0; k<args.dimension; k++){
        pos_a.val[k] = eigvecs->val[k][ pos_edges[j].node_a ] * eigvals_scl[k];
        pos_b.val[k] = eigvecs->val[k][ pos_edges[j].node_b ] * eigvals_scl[k];
        neg_a.val[k] = eigvecs->val[k][ neg_edges[j].node_a ] * eigvals_scl[k];
        neg_b.val[k] = eigvecs->val[k][ neg_edges[j].node_b ] * eigvals_scl[k];
      }

      metric +=  pearson_corr( pos_a, pos_b );
      metric -=  pearson_corr( neg_a, neg_b );

      d_vec_free(pos_a);
      d_vec_free(pos_b);
      d_vec_free(neg_a);
      d_vec_free(neg_b);
    }

    if(metric > best ){
      best_k = i;
      best = metric;  
    } 
  }

  printf("\n\nOptimal step for this split is: %d \n\n",(int) (best_k +1));

  theta[best_k] = 1.0f;
}



//Compute coefficients using constrained hinge loss (SVM-like) : sum{ max(0, 1 - y * c^T * theta ) }
// from prediction of removed (pos_edges) edges and non-existing (neg_edges) ones
// Assumes first half of C corresponds to pos_edges
void svm_fit( d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args ){

  double* C = build_features( eigvecs, eigvals, pos_edges, neg_edges, args);

  sz_med K = args.walk_length;
  double epsilon = HINGE_EPSILON;
	double x_prev[K];
  double x[K];
  double gradient[K];
  double p[2*args.edges_per_split];
  int total_err;

	//Initialize to uniform
	for(sz_med k=0; k<K; k++) x[k]=1.0f/(double)K;

	sz_med iter=0;
	memcpy(x_prev,x,K*sizeof(double));
	do{
		iter++;

    double step_size = STEPSIZE_SVM / (double) sqrt(iter) ;

		//Take gradient step
		matvec(p, C , x, 2*args.edges_per_split, K );

    total_err = 0;
    for(sz_med i =0;i<args.edges_per_split;i++){
      double err = epsilon - p[i];
      p[i] = ( err >= 0.0f ) ? 1.0 : 0.0f;
      total_err += (int) p[i];
    }
    for(sz_med i =args.edges_per_split;i<2*args.edges_per_split;i++){
      double err = epsilon + p[i];
      p[i] = ( err >= 0.0f ) ? -1.0 : 0.0f;
      total_err += (int) -p[i];
    }

    printf("err: %d \n", total_err);

    for(sz_med k =0; k<K; k++){
        gradient[k] = 0.0;
        for(sz_med i =0;i<2*args.edges_per_split;i++)
          gradient[k] += C[i*K + k]*p[i];
    }

    double a_1 = 1.0 - 2.0*step_size*args.lambda;
    double a_2 = step_size/ (double) 2*args.edges_per_split;

    for(sz_med k =0; k<K; k++)
      x[k] = a_1 * x[k] + a_2 * gradient[k];

    //project to feasible set
    if(args.simplex){
        project_to_simplex( x , K );
    }else{
        project_to_pos_quad( x, K );
    }

		memcpy(x_prev,x,K*sizeof(double));

	}while( iter<MAXIT_GD &&  detect_change(total_err) );

	printf("\n Optimization finished after: %"PRIu32" iterations\n", (uint32_t) iter);

  #if PRINT_THETAS
    printf("\ntheta: ");
    print_array(x, 1, K);
    printf("\n");
  #endif

  memcpy(theta,x,K*sizeof(double));

  free(C);
}


//Compute coefficients using simplex constrained logistic regression || b - C * theta  ||_2
// from prediction of removed (pos_edges) edges and non-existing (neg_edges) ones
// Assumes first half of C corresponds to pos_edges
void logistic_fit( d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args ){

  double* C = build_features( eigvecs, eigvals, pos_edges, neg_edges, args);

  sz_med K = args.walk_length;
  double  inf_norm, step_size;
	double x_prev[K];
  double x[K];
  double gradient[K];
  double p[2*args.edges_per_split];

  step_size = STEPSIZE_LOG;


	//Initialize to uniform
	for(sz_med k=0; k<K; k++) x[k]=1.0f/(double)K;


	sz_med iter=0;
	memcpy(x_prev,x,K*sizeof(double));
	do{
		iter++;

		//Take gradient step
		matvec(p, C , x, 2*args.edges_per_split, K );

    for(sz_med i =0;i<args.edges_per_split;i++)
      p[i] = 1.0 / (1.0 + exp(p[i]));
    for(sz_med i =args.edges_per_split;i<2*args.edges_per_split;i++)
      p[i] = -1.0 / (1.0 + exp(-p[i]));


    for(sz_med k =0; k<K; k++){
        gradient[k] = 0.0;
        for(sz_med i =0;i<2*args.edges_per_split;i++)
          gradient[k] += C[i*K + k]*p[i];
    }

    double a_1 = 1.0 - 2.0*step_size*args.lambda;
    double a_2 = step_size/ (double) 2*args.edges_per_split;

    for(sz_med k =0; k<K; k++)
      x[k] = a_1 * x[k] + a_2 * gradient[k];


    //project to feasible set
    if(args.simplex){
        project_to_simplex( x , K );
    }else{
        project_to_pos_quad( x, K );
    }

		inf_norm = max_abs_dif(x_prev,x , (sz_long)K );

		memcpy(x_prev,x,K*sizeof(double));

	}while( iter<MAXIT_GD &&  inf_norm>GD_TOL_2 );

	printf("\n Optimization finished after: %"PRIu32" iterations\n", (uint32_t) iter);

  #if PRINT_THETAS
    printf("\ntheta: ");
    print_array(x, 1, K);
    printf("\n");
  #endif

  memcpy(theta,x,K*sizeof(double));

  free(C);
}



//Compute coefficients using simplex constrained quadratic cost || b - C * theta  ||_2
// from prediction of removed (pos_edges) edges and non-existing (neg_edges) ones
// Assumes first half of C corresponds to pos_edges
void ls_fit( d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args )
{

  double* C = build_features( eigvecs, eigvals, pos_edges, neg_edges, args);

  //Compute coefficients of quadratic cost
  double A[args.walk_length*args.walk_length];
  double b[args.walk_length];

  //A = C'*C
  grammian_matrix(A,C, (int) 2*args.edges_per_split, (int) args.walk_length );

  for(sz_med i=0; i<args.walk_length; i++ ) A[i*args.walk_length + i] += args.lambda;

  //b = -2* y'*case
  for(sz_med i=0; i<args.walk_length; i++ ){
    b[i] = 0.0f;

    //Top half of array C for edges that are present
    for(sz_med j=0; j<args.edges_per_split; j++ )
      b[i] -= C[j*args.walk_length + i];

    //Bottom half of array C for edges that are not present
    for(sz_med j = args.edges_per_split; j<2*args.edges_per_split; j++ )
      b[i] += C[j*args.walk_length + i];

    b[i] *= 2.0f;
  }

  //Solve the simplex constrained QP
  constr_QP_with_PG(theta, A, b, args.walk_length, args.simplex);

  #if DEBUG
    printf("\nA: ");
    print_array(A, (int) args.walk_length, (int) args.walk_length);
    printf("\nb: ");
    print_array(b, 1, (int) args.walk_length);
  #endif

  #if PRINT_THETAS
  printf("\ntheta: ");
  print_array(theta, 1, (int) args.walk_length);
  printf("\n");
  #endif

  free(C);

}


// For methods that require optimization
// This step simplifies the process by intriducing the intermediate C matrix 
static double* build_features(d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                           edge_tuple* neg_edges, cmd_args args)
{
  //Build matrix X of entrywise products
  double* X = (double*) malloc(2*args.edges_per_split*args.dimension*sizeof(double));
  for(sz_med i=0; i<args.edges_per_split; i++){
    sz_long node_a = pos_edges[i].node_a;
    sz_long node_b = pos_edges[i].node_b;
    for(sz_med j=0; j<args.dimension; j++)
      X[i*args.dimension + j] = eigvecs->val[j][node_a]*eigvecs->val[j][node_b];
  }

  for(sz_med i=0; i<args.edges_per_split; i++){
    sz_long node_a = neg_edges[i].node_a;
    sz_long node_b = neg_edges[i].node_b;
    for(sz_med j=0; j<args.dimension; j++)
      X[(i+args.edges_per_split)*args.dimension + j] = eigvecs->val[j][node_a]*eigvecs->val[j][node_b];
  }

  //Build matrix S of eigenvalue powers
  double S[ args.dimension * args.walk_length ];
  for(sz_med i=0; i<args.dimension; i++){
    for(sz_med j=0; j<args.walk_length; j++)
      S[i*args.walk_length + j] = pow(eigvals->val[i], j+1);
//      S[i*args.walk_length + j] = (i==j) ? 1.0f : 0.0;
  }

  //Intermediate matrix C = X* S
  double* C = (double*) malloc(2*args.edges_per_split*args.walk_length*sizeof(double));

  matrix_matrix_product(C, X, S, (int) 2*args.edges_per_split,
                        (int) args.dimension, (int) args.walk_length);

  free(X);

  return C;
}

//Solving constrained quadratic minimization via projected gradient descent
//Constrains are either probability simplex (simplex = TRUE) or positive quadrant (simplex = FALSE)
//Used by AdaDIF and AdaDIF_LOO
// The following function returns x =arg min {x^T*A*x +x^T*B} s.t. x in Prob. Simplex
static void constr_QP_with_PG(double* x, double* A, double* b, sz_med K, bool simplex){
	double  inf_norm, step_size;
	double x_temp[K];
	double x_prev[K];

//	step_size = STEPSIZE_2;
//	step_size = STEPSIZE_2/pow(frob_norm(A,K), 2.0f);
	step_size = STEPSIZE_2/frob_norm(A,K);

	//Initialize to uniform
	for(sz_med i=0;i<K;i++) x[i]=1.0f/(double)K;

	sz_med iter=0;
	memcpy(x_prev,x,K*sizeof(double));
	do{
		iter++;

		//Take gradient step
		matvec(x_temp, A , x, K, K );

		for(sz_med j=0;j<K;j++)
			x[j]-= step_size*( 2.0f*x_temp[j] +b[j] );

    //project to feasible set
    if(simplex){
        project_to_simplex( x , K );
    }else{
        project_to_pos_quad( x, K );
    }


    #if DEBUG
		  printf("\n COST: ");
		  printf(" %lf ",cost_func(A,b,x,K));
		#endif

		inf_norm = max_abs_dif(x_prev,x , (sz_long)K );

		memcpy(x_prev,x,K*sizeof(double));

	}while( iter<MAXIT_GD &&  inf_norm>GD_TOL_2 );

	printf("\n Optimization finished after: %"PRIu32" iterations\n", (uint32_t) iter);

}



// Grammian matrix G =A'*A using CBLAS
// A : m x n
static void grammian_matrix(double* G, double* A, int m, int n ){

  for(int i=0;i<n*n;i++) G[i] = 0.0f;

  double* A_copy = (double*) malloc(m*n*sizeof(double));

  memcpy(A_copy, A, m*n*sizeof(double) );

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n,
		          n, m, 1.0f, A_copy, n, A, n, 0.0f, G, n);

  free(A_copy);
}


//frobenious norm of double-valued square matrix
static double frob_norm(double* A, sz_med dim){
	double norm=0.0f;

	for(sz_med i=0;i<dim*dim;i++){
		norm+=pow(A[i],2.0f);
	}

	return sqrt(norm);
}

//Interface for CBLAS matrix vector product
// A : M x N
static void matvec(double*y, double* A, double* x, int M, int N ){

	for(int i=0;i<M;i++){y[i]=0.0f;}

	cblas_dgemv( CblasRowMajor , CblasNoTrans , M , N, 1.0f, A, N, x, 1, 0.0f, y, 1);


}

//Project vector onto simplex by alternating projections onto line and positive quadrant
//Operation happens in place
static void project_to_simplex( double* x, sz_med N ){
	double sum,a;
	sz_short flag;

	do{
		flag=0;
		sum=0.0f;

		for(sz_med i=0; i<N; i++) sum+=x[i];

		a=(sum - 1.0f)/(double)N;

		for(sz_med i=0; i<N; i++){
			x[i]-=a;
			if(x[i]<= - PROJ_TOL){
				x[i]=0.0f;
				flag=1;}
		}

	}while(flag==1);
}


//Project vector x onto positive quadrant (truncate neg entries to 0.0)
//Operation happens in place
static void project_to_pos_quad(double* x , sz_med N){

  for(sz_med i=0; i<N; i++) x[i] = (x[i] >= 0.0f ) ? x[i] : 0.0 ;

}


//Evaluates quadtratic with Hessian A and linear part b at x

static double cost_func(double* A, double* b, double* x, sz_med len){

	double quad =0.0f, lin = 0.0f;

	for(sz_med i=0;i<len;i++){
		for(sz_med j=0;j<len;j++){
			quad+= A[i*len + j]*x[i]*x[j];
		}
		lin+=b[i]*x[i];
	}
	return quad + lin;
}


//Infinity norm

static double max_abs_dif(double* a, double* b, sz_long len ){
	double dif,max=0.0;

	for(sz_long i=0;i<len;i++){
		dif = fabs(a[i]-b[i]);
		max = (dif>max) ? dif : max ;
	}

	return max;
}




// Returns TRUE if there a change larger than a given threshold
// between any two consequtive inputs. 
// Forgets changes after N inputs

bool detect_change( int x ){

  bool change = true;
  
  static int last_input = 0;
  static int time_since_last_change = 0;

  if(abs(x-last_input) > DC_THRES ){
    time_since_last_change = 0;
  }else{
    time_since_last_change ++;
  }

  if(time_since_last_change > DC_WIN_LEN)
    change = false; 

  last_input = x;

  return change;
}


//Interface for CBLAS mutrix matrix product
// C =A*B
// A : m x k
// B : k x n
static void matrix_matrix_product(double* C, double* A, double* B, int m, int k , int n){

  for(int i=0;i<m*n;i++) C[i] = 0.0f;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,
		          n, k, 1.0f, A, k, B, n, 0.0f, C, n);

}


//Pearson correlation between two vectors
double pearson_corr(d_vec a, d_vec b){

  assert(a.num_entries == b.num_entries);

  double corr, dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

  int N = a.num_entries;

  for(int i=0; i<N; i++){
    dot += a.val[i]*b.val[i];
    norm_a += a.val[i]*a.val[i];
    norm_b += b.val[i]*b.val[i];
  }

  corr = dot / sqrt( norm_a * norm_b );

return corr;
}



