# Copyright 2019, Julius Huijts
# Authors: Julius Huijts
# License: BSD-3-Clause
"""
This file is part of the numerical monochromatization method for the 
treatment of broadband diffraction patterns. It contains the functions 
needed for monochromatization procedure.
"""


import numpy as np
import scipy.sparse as sparse
from joblib import Parallel, delayed
from Cparloop_sparse import parloop
import scipy.ndimage as ndimage

def build_C(S, a, Npixels, n_jobs=-2, verbose=10, Cmode='analysis'):
    ## Builds matrix C
    ## Cmode = 'analysis' or 'generation'
    #if __name__ == '__main__':
    listofCblocks = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(parloop)(Npixels, a, S, k_x, Cmode) for k_x in range(1,Npixels+1))
    
    C_csr = sparse.vstack(listofCblocks,format='csr') # C in sparse csr format
    C_full_csr = sparse.kron(np.eye(4),C_csr,format='csr')
    
    return C_full_csr

def cut_rotate_ravel(B):
    ## Prepares blurred pattern B for the monochromatization by matrix C
    Npixels = int(np.sqrt(B.size)/2)
    # cut image in quadrants
    Bquad1 = B[0:Npixels, 0:Npixels]
    Bquad2 = B[0:Npixels, Npixels:2*Npixels]
    Bquad3 = B[Npixels:2*Npixels, Npixels:2*Npixels]
    Bquad4 = B[Npixels:2*Npixels, 0:Npixels]

    # rotate quadrants, ravel and append for deblurring 
    B_ravel = np.zeros(4*Npixels**2)
    B_ravel[0:Npixels**2] = np.rot90(Bquad1, -2).ravel('F')
    B_ravel[Npixels**2:2*Npixels**2] = np.rot90(Bquad2, -1).ravel('F')
    B_ravel[2*Npixels**2:3*Npixels**2] = Bquad3.ravel('F')
    B_ravel[3*Npixels**2:4*Npixels**2] = np.rot90(Bquad4, 1).ravel('F')
    
    return B_ravel


def inverse_ravel_rotate_cut(mk_2D_ravel):
    Npixels = int(np.sqrt(mk_2D_ravel.size)/2)
    
    mk_2D = np.zeros((2*Npixels, 2*Npixels))
    mk_2D[0:Npixels, 0:Npixels] = np.rot90(mk_2D_ravel[0:Npixels**2].reshape((Npixels,Npixels),order='F'), 2)
    mk_2D[0:Npixels, Npixels:2*Npixels] = np.rot90(mk_2D_ravel[Npixels**2:2*Npixels**2].reshape((Npixels,Npixels),order='F'), 1)
    mk_2D[Npixels:2*Npixels, Npixels:2*Npixels] = mk_2D_ravel[2*Npixels**2:3*Npixels**2].reshape((Npixels,Npixels),order='F')
    mk_2D[Npixels:2*Npixels, 0:Npixels] = np.rot90(mk_2D_ravel[3*Npixels**2:4*Npixels**2].reshape((Npixels,Npixels),order='F'), -1)
    
    return mk_2D

def CGLS_sparse(A,b,k_max=30,reorth=True,nonneg=True):
    # As in the Matlab function by P.C. Hansen
    # X = RRGMRES(A,b,k_max)
    # but for a sparse A in csr format
    
    reorth = bool(reorth)
    
    if sparse.isspmatrix_csr(A)==False:
        A = A.tocsr()
    
    A_shape0 = A.shape[0]
    A_shape1 = A.shape[1]

    # Allocate space
    X = np.zeros((A_shape1,k_max)) # Matrix of solutions.
    if reorth:
        ATr = np.zeros((A_shape1,k_max+1))
    x = np.zeros(A_shape1)
    d = A.transpose()*b
    r = b.copy()
    normr2 = np.inner(d,d)
    if reorth:
        ATr[:,0] = d/np.linalg.norm(d)
    
    # Iterate
    for j in range(k_max):
        
        # Update x and r vectors
        Ad = A*d
        alpha = normr2/np.inner(Ad,Ad)
        x_new = x + alpha*d
        if nonneg:
            x_new_nonneg = abs(x_new*(x_new>0))
            x[:] = x_new_nonneg[:]
        else:
            x[:] = x_new[:]
        r_new = r - alpha*Ad
        r[:] = r_new[:]
        s = A.transpose()*r
        
        # Reorthogonalize s to previous s vectors
        if reorth:
            for i in range(j):
                s_new = s - np.inner(ATr[:,i],s)*ATr[:,i]
                s[:] = s_new[:]
            ATr[:,j+1] = s/np.linalg.norm(s)
            
        # Update d vector
        normr2_new = np.inner(s,s)
        beta = normr2_new/normr2
        normr2 = normr2_new.copy()
        d_new = s + beta*d
        d[:] = d_new[:]
        
        X[:,j] = x
        
    return X


def CGLS_sparse_supcon(A,b,supfilter,k_max=30,reorth=True,nonneg=True):
    # As in the Matlab function by P.C. Hansen
    # X = RRGMRES(A,b,k_max)
    # but for a sparse A in csr format
    # and taking into account a support constraint in real space
    
    reorth = bool(reorth)
    nonneg = bool(nonneg)
    
    if sparse.isspmatrix_csr(A)==False:
        A = A.tocsr()
    
    A_shape0 = A.shape[0]
    A_shape1 = A.shape[1]

    # Allocate space
    X = np.zeros((A_shape1,k_max)) # Matrix of solutions.
    if reorth:
        ATr = np.zeros((A_shape1,k_max+1))
    x = np.zeros(A_shape1)
    d = A.transpose()*b
    r = b.copy()
    normr2 = np.inner(d,d)
    if reorth:
        ATr[:,0] = d/np.linalg.norm(d)
    
    # Iterate
    for j in range(k_max):
        
        # Update x and r vectors
        Ad = A*d
        alpha = normr2/np.inner(Ad,Ad)
        x_new = x + alpha*d
        if nonneg:
            x_new_nonneg = abs(x_new*(x_new>0))
            x[:] = x_new_nonneg[:]
        else:
            x[:] = x_new[:]
        x_2d = inverse_ravel_rotate_cut(x)
        x_2d_FT = np.fft.fft2(x_2d)
        x_2d_FT_filtered = x_2d_FT*supfilter
        x_2d_filtered = abs(np.fft.ifft2(x_2d_FT_filtered))
        x_filtered = cut_rotate_ravel(x_2d_filtered)
        x = x_filtered.copy()
        
        r_new = r - alpha*Ad
        r[:] = r_new[:]
        s = A.transpose()*r
        
        # Reorthogonalize s to previous s vectors
        if reorth:
            for i in range(j):
                s_new = s - np.inner(ATr[:,i],s)*ATr[:,i]
                s[:] = s_new[:]
            ATr[:,j+1] = s/np.linalg.norm(s)
            
        # Update d vector
        normr2_new = np.inner(s,s)
        beta = normr2_new/normr2
        normr2 = normr2_new.copy()
        d_new = s + beta*d
        d[:] = d_new[:]
        
        X[:,j] = x
        
    return X

def build_C_1D(Npixels, a, S, mode):
    ## For illustration
    N = np.arange(1, Npixels+1)
    C = sparse.lil_matrix((Npixels, Npixels), dtype=np.float32)
    
    if mode == 'generation':
        
        for k_x in range(1,Npixels+1):
            n_x_min = (-1*(N > (k_x-1)/max(a))+1).sum()+1
            n_x_max = ((N < (k_x/min(a)+1))*1).sum()
            for n_x in range(n_x_min,n_x_max+1):
                if (n_x != 1):
                    a_l = a[(a > (k_x-1)/n_x) & (a < k_x/(n_x-1))]
                    S_L = S[(a > (k_x-1)/n_x) & (a < k_x/(n_x-1))]
                else:
                    a_l = a[(a > (k_x-1)/n_x)]
                    S_L = S[(a > (k_x-1)/n_x)]

                f_k_xn_xl = (np.minimum(a_l*n_x,k_x*np.ones_like(a_l))-np.maximum(a_l*(n_x-1),(k_x-1)*np.ones_like(a_l)))/a_l
                C[k_x-1, (n_x-1)] = np.dot(S_L, f_k_xn_xl).astype('float32')
                
    if mode == 'analysis':
        
        for k_x in range(1,Npixels+1):
            n_x_min = (-1*(N > (k_x-1)/max(a))+1).sum()+1
            n_x_max = np.minimum(((N < (k_x/min(a)+1))*1).sum(), Npixels)
            for n_x in range(n_x_min,n_x_max+1):
                if (n_x != 1):
                    a_l = a[(a > (k_x-1)/n_x) & (a < k_x/(n_x-1))]
                    S_L = S[(a > (k_x-1)/n_x) & (a < k_x/(n_x-1))]
                else:
                    a_l = a[(a > (k_x-1)/n_x)]
                    S_L = S[(a > (k_x-1)/n_x)]

                f_k_xn_xl = (np.minimum(a_l*n_x,k_x*np.ones_like(a_l))-np.maximum(a_l*(n_x-1),(k_x-1)*np.ones_like(a_l)))/a_l
                C[k_x-1, (n_x-1)] = np.dot(S_L, f_k_xn_xl).astype('float32')
    return C


def CGLS_sparse_supcon_1D(A,b,supfilter,k_max=30,reorth=True,nonneg=True):
    ## For illustration. Assumes that pixel 0 is in the center, so supfilter.size should be 2*Npixels-1
    
    reorth = bool(reorth)
    nonneg = bool(nonneg)
    
    if sparse.isspmatrix_csr(A)==False:
        A = A.tocsr()
    
    A_shape0 = A.shape[0]
    A_shape1 = A.shape[1]

    # Allocate space
    X = np.zeros((A_shape1,k_max)) # Matrix of solutions.
    if reorth:
        ATr = np.zeros((A_shape1,k_max+1))
    x = np.zeros(A_shape1)
    d = A.transpose()*b
    r = b.copy()
    normr2 = np.inner(d,d)
    if reorth:
        ATr[:,0] = d/np.linalg.norm(d)
    
    # Iterate
    for j in range(k_max):
        
        # Update x and r vectors
        Ad = A*d
        alpha = normr2/np.inner(Ad,Ad)
        x_new = x + alpha*d
        if nonneg:
            x_new_nonneg = abs(x_new*(x_new>0))
            x[:] = x_new_nonneg[:]
        else:
            x[:] = x_new[:]
            
        x_full = np.hstack((x[:0:-1],x))
        x_full_FT = np.fft.fftshift(np.fft.fft(x_full))
        x_full_FT_filtered = x_full_FT*supfilter
        x_full_filtered = abs(np.fft.ifft(np.fft.ifftshift(x_full_FT_filtered)))
        x_filtered = x_full_filtered[(x.size-1):]
        x = x_filtered.copy()
        
        r_new = r - alpha*Ad
        r[:] = r_new[:]
        s = A.transpose()*r
        
        # Reorthogonalize s to previous s vectors
        if reorth:
            for i in range(j):
                s_new = s - np.inner(ATr[:,i],s)*ATr[:,i]
                s[:] = s_new[:]
            ATr[:,j+1] = s/np.linalg.norm(s)
            
        # Update d vector
        normr2_new = np.inner(s,s)
        beta = normr2_new/normr2
        normr2 = normr2_new.copy()
        d_new = s + beta*d
        d[:] = d_new[:]
        
        X[:,j] = x
        
    return X

