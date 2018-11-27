#ifndef TRAIN_H_
#define TRAIN_H_

#include "emb_defs.h"

void  single_best(d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args );

void  ls_fit(d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args );

void  svm_fit(d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args );

void  logistic_fit(d_mat* eigvecs, d_vec* eigvals, edge_tuple* pos_edges,
                  edge_tuple* neg_edges, double* theta, cmd_args args );

#define NUM_METHODS 4

#endif
