import numpy as np 
import matplotlib.pyplot as plt
import argparse
import warnings
from matplotlib2tikz import save as tikz_save

warnings.filterwarnings("error")

def gen_graph(M,N):
    A = np.zeros([N,N])
    for i in range(N):
        for j in range(i+1,N):
            if np.random.rand() < M[i,j]:
                A[i,j] = 1.0
            A[j,i] = A[i,j]    
    return A


def get_ideal(p,q,N):
    A = np.zeros([N,N])
    for i in range(N):
        for j in range(i+1,N):
            if (i<N//3 and j<N//3) or (i in range(N//3,2*(N//3)) and j in range(N//3, 2*(N//3))) or (i in range(2*(N//3),N) and j in range(2*(N//3),N)):
                A[i,j] = p
            else:
                if i<N/3 and j>2*(N//3) or j<N/3 and i>2*(N//3):
                    A[i,j] = q/3
                else:
                    A[i,j] = q	
            A[j,i] = A[i,j]    
    return A    



def get_powers(A,K):
    seq = []

    # normalize and shift first
    degrees = np.sum(A,axis = 0)
    if np.sum( degrees == 0.0 ) > 0:
        return seq
    d_inv = degrees**(-0.5)
    D = np.diag(d_inv)    
    P = 0.5*(np.eye(A.shape[0]) + np.dot(np.dot(D,A),D))

    seq = [ P ]
    for k in range(1,K):
        seq.append(np.dot(seq[k-1], P))

    return seq    

def get_adj_powers(A,K):
    seq = [ A ]
    for k in range(1,K):
        seq.append(np.dot(seq[k-1], A))
    return seq    

def get_pgr(A,a):
    degrees = np.sum(A,axis = 0)
    d_inv = degrees**(-1)
    D = np.diag(d_inv) 
    A = np.eye(A.shape[0]) - a*np.dot(A,D)
    A_pgr = np.linalg.inv(A)
    for i in range(A.shape[0]):
        A_pgr[i,i] = 0.0
    #plt.matshow(A_pgr)
    #plt.show()
    return A_pgr

def get_katz(A,beta):
    A_temp = np.eye(A.shape[0]) - beta*A
    A_temp2 = np.linalg.inv(A_temp)
    A_katz = beta*np.dot(A_temp2,A)
    for i in range(A.shape[0]):
        A_katz[i,i] = 0.0    
    # plt.matshow(A_katz)
    # plt.show()
    return A_katz

def get_neigh(A):
    A_neigh = np.dot(A,A)
    for i in range(A.shape[0]):
        A_neigh[i,i] = 0.0
    # plt.matshow(A_neigh)
    # plt.show()    
    return A_neigh    

def get_adam(A,a):
    degrees = np.sum(A, axis = 0)
    D = np.diag(degrees**(-1))
    A_adam = np.dot(np.dot(A,D),A)
    for i in range(A.shape[0]):
        A_adam[i,i] = 0.0    
    #plt.matshow(A_adam)
    #plt.show()
    return A_adam    

def relative_error(A,B):
    # in case you want to center
    # even though it doesnt make much sense in this case
    N = A.shape[0]
    A_norm = A - np.sum(np.sum(A))/(N**2 - N)
    B_norm = B - np.sum(np.sum(B))/(N**2 - N)
    return pearson(A_norm,B_norm)

def pearson(A,B):
    cor = np.sum(np.sum(A*B))
    cor /= np.linalg.norm(A,ord='fro')*np.linalg.norm(B,ord='fro')
    return cor


def main():    
    parser = argparse.ArgumentParser(description='Input SBM parameters p and q')
    parser.add_argument('--p', type=float, default = 0.3 )
    parser.add_argument('--q', type=float, default = 0.1 )
    args = parser.parse_args()


    iters = 100
    N = 150
    K = 30
      
    A_ideal = get_ideal(args.p,args.q,N)
    
    #plt.matshow(A_ideal)
    #plt.show()

    a = 0.95
    beta = 0.04

    error = np.zeros([K,])
    error2 = np.zeros([K,])
    error_pgr = 0.0
    error_katz = 0.0
    error_neigh = 0.0
    error_adam = 0.0
    suc_iters = 0

    for _iter in range(iters):
        A = gen_graph(A_ideal,N)
        seq = get_powers(A,K)
        seq2 = get_adj_powers(A,K)
        if seq:
            suc_iters +=1 
            for k,P_k,A_k in zip(range(K),seq,seq2):
                for i in range(N):
                    A_k[i,i] = 0.0
                    P_k[i,i] = 0.0 
                #plt.matshow(A_k)
                #plt.show()    
                error[k] += pearson(A_ideal,P_k)
                error2[k] += pearson(A_ideal,A_k)
            A_pgr = get_pgr(A,a)
            A_katz = get_katz(A,beta)
            A_neigh = get_neigh(A)
            A_adam = get_adam(A,a)
            error_pgr += pearson(A_ideal,A_pgr)     
            error_katz += pearson(A_ideal,A_katz)   
            error_neigh += pearson(A_ideal,A_neigh)   
            error_adam += pearson(A_ideal,A_adam)                                       

    plt.plot(error/suc_iters, label="$\mathbf{S}^k$" )
    plt.plot(error2/suc_iters, label="$\mathbf{A}^k$", marker = 'o' )    
    plt.plot( (error_pgr/suc_iters) * np.ones([K,]), label = "$\mathbf{S}_{PGR}$", marker='+' )
    plt.plot( (error_katz/suc_iters) * np.ones([K,]),label = "$\mathbf{S}_{KATZ}$", marker = '*' )
    plt.plot( (error_neigh/suc_iters) * np.ones([K,]),label = "$\mathbf{S}_{NEIGH}$", marker = '>' )
    plt.plot( (error_adam/suc_iters) * np.ones([K,]),label = "$\mathbf{S}_{AA}$", marker = '^' )
    plt.xlabel("k")
    plt.legend()
    tikz_save("../figs/p_" + str(int(100*args.p)) + "_q_"+ str(int(100*args.q)) +".tex")
    plt.show()

if __name__ == '__main__':
    main()
