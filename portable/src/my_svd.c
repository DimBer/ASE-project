///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains EMB wrapper of svdlib

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

#include "emb_defs.h"
#include "my_svd.h"
#include "svdlib.h"


//Wrapper of svdlib using my data structs
void my_svd(csr_graph graph, d_mat* U, d_vec* S, sz_med dimension ){

   double kappa = 1e-6 ;
   double las2end[2] = {-1.0e-30, 1.0e-30};
   long iterations = 2000;

   struct smat A;
   A = (struct smat) {.rows = (long) graph.num_nodes,
                       .cols = (long) graph.num_nodes,
                       .vals = (long) graph.nnz,
                       .pointr = (long*) graph.csr_row_pointer,
                       .rowind = (long*) graph.csr_column,
                       .value = (double*) graph.csr_value };

   SVDRec svd = svdNewSVDRec();

   svd = svdLAS2(&A, dimension, iterations, las2end, kappa);

   for(sz_long i=0;i<dimension;i++) memcpy(U->val[i], svd->Ut->value[i], graph.num_nodes*sizeof(double));
   memcpy(S->val , svd->S, dimension*sizeof(double));

   //check symmetry
   #if DEBUG
   double total_dif=0.0f;
   int dif_count = 0;
   for(sz_long i=0;i<dimension;i++){
       double row_dif=0.0f;
       for(sz_long j=0;j<graph.num_nodes;j++){
         double dif = pow(fabs(svd->Ut->value[i][j] - svd->Vt->value[i][j]),2);
         if(dif > 0.1) dif_count++;
         row_dif += dif;
       }
       total_dif += sqrt(row_dif);
   }
   printf("\n\nTotal Dif = %lf \n at %d points", total_dif, dif_count);

   printf("\nEigenvalues: ");
   for(sz_med i =0; i<dimension ;i++) printf(" %lf ",svd->S[i]);

   printf("\n\n");
   #endif

   svdFreeSVDRec(svd);

}
