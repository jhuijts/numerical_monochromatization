# Copyright 2019, Julius Huijts
# Authors: Julius Huijts
# License: BSD-3-Clause
"""
This file is part of the numerical monochromatization method for the 
treatment of broadband diffraction patterns. It contains the functions 
that generate matrix C
"""


import numpy as np
import scipy.sparse as sparse


def parloop(Npixels, a, S, k_x, mode):
## To generate matrix C in a parallel fashion. The way matrix C is built up 
## when generating a polychromatic diffraction pattern (mode='generation') 
## differs from the way it is built up when analyzing (or monochromatizing) 
## a diffraction pattern (mode='analysis') in the upperlimit on n_max.
    
    N = np.arange(1, Npixels+1)
    C_block = sparse.lil_matrix((Npixels, Npixels**2), dtype=np.float32)
    
    if mode == 'generation':
        
        for k_y in range(1,Npixels+1):
            n_x_min = (-1*(N > (k_x-1)/max(a))+1).sum()+1
            n_y_min = (-1*(N > (k_y-1)/max(a))+1).sum()+1
            n_x_max = ((N < (k_x/min(a)+1))*1).sum()
            n_y_max = ((N < (k_y/min(a)+1))*1).sum()
            for n_x in range(n_x_min,n_x_max+1):
                for n_y in range(n_y_min,n_y_max+1):
                    if (n_x != 1) & (n_y != 1):
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                    elif (n_x == 1) & (n_y != 1):
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_y/(n_y-1))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_y/(n_y-1))]
                    elif (n_x != 1) & (n_y == 1):
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_x/(n_x-1))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_x/(n_x-1))]
                    else:
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
    
                    f_k_xn_xl = (np.minimum(a_l*n_x,k_x*np.ones_like(a_l))-np.maximum(a_l*(n_x-1),(k_x-1)*np.ones_like(a_l)))/a_l
                    f_k_yn_yl = (np.minimum(a_l*n_y,k_y*np.ones_like(a_l))-np.maximum(a_l*(n_y-1),(k_y-1)*np.ones_like(a_l)))/a_l
                    f_k_xk_yn_xn_yl = f_k_xn_xl*f_k_yn_yl
                    C_block[k_y-1, (n_x-1)*Npixels+n_y-1] = np.dot(S_L, f_k_xk_yn_xn_yl).astype('float32')

    if mode == 'analysis':
        for k_y in range(1,Npixels+1):
            n_x_min = (-1*(N > (k_x-1)/max(a))+1).sum()+1
            n_y_min = (-1*(N > (k_y-1)/max(a))+1).sum()+1
            n_x_max = np.minimum(((N < (k_x/min(a)+1))*1).sum(), Npixels)
            n_y_max = np.minimum(((N < (k_y/min(a)+1))*1).sum(), Npixels)
            for n_x in range(n_x_min,n_x_max+1):
                for n_y in range(n_y_min,n_y_max+1):
                    if (n_x != 1) & (n_y != 1):
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                    elif (n_x == 1) & (n_y != 1):
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_y/(n_y-1))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_y/(n_y-1))]
                    elif (n_x != 1) & (n_y == 1):
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_x/(n_x-1))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (a < k_x/(n_x-1))]
                    else:
                        a_l = a[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
                        S_L = S[(a > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
    
                    f_k_xn_xl = (np.minimum(a_l*n_x,k_x*np.ones_like(a_l))-np.maximum(a_l*(n_x-1),(k_x-1)*np.ones_like(a_l)))/a_l
                    f_k_yn_yl = (np.minimum(a_l*n_y,k_y*np.ones_like(a_l))-np.maximum(a_l*(n_y-1),(k_y-1)*np.ones_like(a_l)))/a_l
                    f_k_xk_yn_xn_yl = f_k_xn_xl*f_k_yn_yl
                    C_block[k_y-1, (n_x-1)*Npixels+n_y-1] = np.dot(S_L, f_k_xk_yn_xn_yl).astype('float32')
                    
    if not (mode == 'generation' or mode == 'analysis'):
        print('mode should be either \'generation\' or \'analysis\'')
    
    return C_block

def C_ewaldcorr(Npixels, a, S, k_x, delta, mode):
## To generate matrix C in a parallel fashion, including the difference in Ewald sphere curvature for the different wavelengths. The way matrix C is built up when generating a polychromatic diffraction pattern (mode='generation') differs from the way it is built up when analyzing (or monochromatizing) a diffraction pattern (mode='analysis') in the upperlimit on n_max.
    
    N = np.arange(1, Npixels+1)
    C_block = sparse.lil_matrix((Npixels, Npixels**2), dtype=np.float32)
    
    if mode == 'generation':
        
        for k_y in range(1,Npixels+1):
            k_r = np.sqrt((k_x-1/2)**2 + (k_y-1/2)**2)
            b = (k_r*delta)/(np.tan(2*np.arcsin(1/a*np.sin(1/2*np.arctan(k_r*delta))))) # the Ewald sphere correction of a
            n_x_min = (-1*(N > (k_x-1)/max(b))+1).sum()+1
            n_y_min = (-1*(N > (k_y-1)/max(b))+1).sum()+1
            n_x_max = ((N < (k_x/min(b)+1))*1).sum()
            n_y_max = ((N < (k_y/min(b)+1))*1).sum()
            for n_x in range(n_x_min,n_x_max+1):
                for n_y in range(n_y_min,n_y_max+1):
                    if (n_x != 1) & (n_y != 1):
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                    elif (n_x == 1) & (n_y != 1):
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_y/(n_y-1))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_y/(n_y-1))]
                    elif (n_x != 1) & (n_y == 1):
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_x/(n_x-1))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_x/(n_x-1))]
                    else:
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
    
                    f_k_xn_xl = (np.minimum(b_l*n_x,k_x*np.ones_like(b_l))-np.maximum(b_l*(n_x-1),(k_x-1)*np.ones_like(b_l)))/b_l
                    f_k_yn_yl = (np.minimum(b_l*n_y,k_y*np.ones_like(b_l))-np.maximum(b_l*(n_y-1),(k_y-1)*np.ones_like(b_l)))/b_l
                    f_k_xk_yn_xn_yl = f_k_xn_xl*f_k_yn_yl
                    C_block[k_y-1, (n_x-1)*Npixels+n_y-1] = np.dot(S_L, f_k_xk_yn_xn_yl).astype('float32')

    if mode == 'analysis':
        for k_y in range(1,Npixels+1):
            k_r = np.sqrt((k_x-1/2)**2 + (k_y-1/2)**2)
            b = (k_r*delta)/(np.tan(2*np.arcsin(1/a*np.sin(1/2*np.arctan(k_r*delta))))) # the Ewald sphere correction of a
            n_x_min = (-1*(N > (k_x-1)/max(b))+1).sum()+1
            n_y_min = (-1*(N > (k_y-1)/max(b))+1).sum()+1
            n_x_max = np.minimum(((N < (k_x/min(b)+1))*1).sum(), Npixels)
            n_y_max = np.minimum(((N < (k_y/min(b)+1))*1).sum(), Npixels)
            for n_x in range(n_x_min,n_x_max+1):
                for n_y in range(n_y_min,n_y_max+1):
                    if (n_x != 1) & (n_y != 1):
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < np.minimum(k_x/(n_x-1), k_y/(n_y-1)))]
                    elif (n_x == 1) & (n_y != 1):
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_y/(n_y-1))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_y/(n_y-1))]
                    elif (n_x != 1) & (n_y == 1):
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_x/(n_x-1))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y)) & (b < k_x/(n_x-1))]
                    else:
                        b_l = b[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
                        S_L = S[(b > np.maximum((k_x-1)/n_x, (k_y-1)/n_y))]
    
                    f_k_xn_xl = (np.minimum(b_l*n_x,k_x*np.ones_like(b_l))-np.maximum(b_l*(n_x-1),(k_x-1)*np.ones_like(b_l)))/b_l
                    f_k_yn_yl = (np.minimum(b_l*n_y,k_y*np.ones_like(b_l))-np.maximum(b_l*(n_y-1),(k_y-1)*np.ones_like(b_l)))/b_l
                    f_k_xk_yn_xn_yl = f_k_xn_xl*f_k_yn_yl
                    C_block[k_y-1, (n_x-1)*Npixels+n_y-1] = np.dot(S_L, f_k_xk_yn_xn_yl).astype('float32')
                    
    if not (mode == 'generation' or mode == 'analysis'):
        print('mode should be either \'generation\' or \'analysis\'')
    
    return C_block
