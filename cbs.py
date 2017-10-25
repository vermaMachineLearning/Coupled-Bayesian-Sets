# -*- coding: utf-8 -*-
"""
@authors: Saurabh Verma, Estevam R. Hruschka Jr
"""

import scipy.io
import pandas as pd
import numpy as np
import os


################# Load Binary Feature Matrix in Sparse COmpressed Row Format #######################

mat = scipy.io.loadmat('data/nell_exp_X_transpose.mat')
X=mat['X']
X=X.transpose()

################# Load All Set Elements #######################

df_set=pd.read_csv('data/all_set_elements.csv',sep='<#>',header=None)
df_set.columns=['Set_Element']

################# Compute Each Class Weight Vector Based On Bayesian Sets#######################

data_files=os.listdir('data/')
data_files.sort()


class_weight_vectors=list()
class_biases=list()
class_query_feature_sums=list()

for filename in data_files:
    
    if filename.startswith("class_seed_set"):
        
        print("Reading file "+filename+"...")
        class_seed_set_csv='data/'+filename
        df_class_seed_set=pd.read_csv(class_seed_set_csv,header=None)
        df_class_seed_set.columns=['Set_Element']
        
        #compute class weight vector using bayesian set logic 
        
        query_idx=df_set.loc[df_set['Set_Element'].isin(df_class_seed_set['Set_Element'].tolist())].index.tolist()
        query_idx=np.array(query_idx)
        
        query_feature_matrix=X[query_idx,:]; #pull out the query set feature vectors from the dataset
        query_feature_sum=np.sum(query_feature_matrix,0);
        
        scf=2; #tunable parameter
        
        XM=np.mean(X,0)+1e-12;
        alphap=scf*XM; 
        betap=scf*(1-XM);
        lal=np.log(alphap);
        lbe=np.log(betap);
        lab=np.log(alphap+betap);
        
        N=len(query_idx);
        labn=np.log(alphap+betap+N);
        
        lbp=np.log(betap+N-query_feature_sum);
        class_bias=np.sum(lab-labn+lbp-lbe,1);
        w_class_vec=np.log(alphap+query_feature_sum)-lal-lbp+lbe;
       
        
        #store each class weight vector and biases
                
        class_weight_vectors.append(np.array(w_class_vec).reshape((w_class_vec.shape[1],)))
        class_biases.append(np.array(class_bias[0,0]))
        class_query_feature_sums.append(np.array(query_feature_sum).reshape((query_feature_sum.shape[1],)))


class_query_feature_sums=np.asarray(class_query_feature_sums)
class_weight_vectors=np.asarray(class_weight_vectors)
class_biases=np.asarray(class_biases)

#Penalize class weights according to coupled bayesian set logic

n_classes=class_weight_vectors.shape[0]
class_pscores=list()

for c in range(0, n_classes):
    
    
    curr_class_nz_feature_idx=class_query_feature_sums[c].nonzero()
    curr_w_class_vec=class_weight_vectors[c]
    
    for i in range(0, n_classes):
        
        penalize_w_vec=np.zeros(curr_w_class_vec.shape)
        
        if (i != c):
            
            nz_feature_idx=class_query_feature_sums[i].nonzero()
            penalize_idx=np.setdiff1d(nz_feature_idx, curr_class_nz_feature_idx)
            penalize_w_vec[penalize_idx]=class_weight_vectors[i][penalize_idx]
            
            curr_w_class_vec=curr_w_class_vec-penalize_w_vec

    score=np.sum(X.multiply(curr_w_class_vec),1)
    pscore=score+class_biases[c];
    class_pscores.append(np.array(pscore).reshape((pscore.shape[0],)))
    
class_pscores=np.asarray(class_pscores)


#Print top k mutually exclusive set elements for each class
#You can also normalize score between [0,1] before expanding the set

top_k=50    

for c in range(0, n_classes):
    
    curr_mutual_excl_elem=np.ones(class_pscores.shape[1], dtype=bool)
    
    for i in range(i, n_classes):
        
        if (i != c): 
            curr_mutual_excl_elem=np.logical_and((class_pscores[c]>class_pscores[i]),curr_mutual_excl_elem)

    mutual_excl_elem_idx=np.where(curr_mutual_excl_elem)[0]
    pscore=class_pscores[c][mutual_excl_elem_idx]
    df_set_temp=df_set.iloc[mutual_excl_elem_idx]
    top_idx=pscore.argsort()[::-1][:top_k]
    print(df_set_temp.iloc[top_idx])
    print('########################')
       


    
    
    