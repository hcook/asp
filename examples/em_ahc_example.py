import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit
import copy

from em import *

def ahc_using_2sec_voting(gmm_list, k, X):
    # Performs agglomerative hierarchical clustering on the GMMs contained in gmm_list
    # X must be a 2-dimensional numpy.ndarray  
    
    N = X.shape[0]
    D = X.shape[1]

    # Get the events, divide them into an initial k clusters and train each GMM on a cluster
    per_cluster = N / k
    init_training = zip(gmm_list,np.vsplit(X, range(per_cluster, N, per_cluster)))
    for g, x in init_training:
        g.train(x)

    # Perform hierarchical agglomeration based on BIC scores
    best_BIC_score = 1.0
    while (best_BIC_score > 0 and len(gmm_list) > 1):
        print "Num GMMs: %d, last score: %d" % (len(gmm_list), best_BIC_score)

        num_clusters = len(gmm_list)
        # Resegment data based on likelihood scoring
        likelihoods = gmm_list[0].score(X)
        for g in gmm_list[1:]:
            likelihoods = np.column_stack((likelihoods, g.score(X)))
        most_likely = likelihoods.argmax(axis=1)
        # Across 2.5 secs of observations, vote on which cluster they should be associated with
        iter_training = {}
        for i in range(250, N, 250):
            votes = np.zeros(num_clusters)
            for j in range(i-250, i):
                votes[most_likely[j]] += 1
            iter_training.setdefault(gmm_list[votes.argmax()],[]).append(X[i-250:i,:])
        # Handle remainder if N % 250 != 0
        votes = np.zeros(num_clusters)
        for j in range((N/250)*250, N):
            votes[most_likely[j]] += 1
        iter_training.setdefault(gmm_list[votes.argmax()],[]).append(X[(N/250)*250:N,:])

        # Retrain the GMMs on the clusters for which they were voted most likely and
        # make a list of candidates for merging
        iter_bic_list = []
        for g, data_list in iter_training.iteritems():
            cluster_data =  data_list[0]
            for d in data_list[1:]:
                cluster_data = np.concatenate((cluster_data, d))
            cluster_data = np.ascontiguousarray(cluster_data)
            g.train(cluster_data)
            iter_bic_list.append((g,cluster_data))

        # Keep any GMMs that lost all votes in candidate list for merging
        for g in gmm_list:
            if g not in iter_training.keys():
                iter_bic_list.append((g,None))            

        # Score all pairs of GMMs using BIC
        best_merged_gmm = None
        best_BIC_score = 0.0
        merged_tuple = None
        for gmm1idx in range(len(iter_bic_list)):
            for gmm2idx in range(gmm1idx+1, len(iter_bic_list)):
                g1, d1 = iter_bic_list[gmm1idx]
                g2, d2 = iter_bic_list[gmm2idx] 
                score = 0.0
                if d1 is not None or d2 is not None:
                    if d1 is not None and d2 is not None:
                        new_gmm, score = compute_distance_BIC(g1, g2, np.concatenate((d1, d2)))
                    elif d1 is not None:
                        new_gmm, score = compute_distance_BIC(g1, g2, d1)
                    else:
                        new_gmm, score = compute_distance_BIC(g1, g2, d2)
                print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                if score > best_BIC_score: 
                    best_merged_gmm = new_gmm
                    merged_tuple = (g1, g2)
                    best_BIC_score = score
        
        # Merge the winning candidate pair if its deriable to do so
        if best_BIC_score > 0.0:
            gmm_list.remove(merged_tuple[0]) 
            gmm_list.remove(merged_tuple[1]) 
            gmm_list.append(best_merged_gmm)

        print "Final size of each cluster:", [ g.M for g in gmm_list]
        
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: ahc_example.py initial_components_per_cluster intial_num_clusters input_data.csv" 
        sys.exit()
    X = np.ndfromtxt(sys.argv[3], delimiter=',', dtype=np.float32)
    N = X.shape[0]
    D = X.shape[1]
    M = int(sys.argv[1])
    init_num_clusters = int(sys.argv[2])
    gmm_list = [GMM(M, D) for i in range(init_num_clusters)]
    ahc_using_2sec_voting(gmm_list, init_num_clusters, X)

