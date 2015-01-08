import scipy.cluster.vq as vq
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

def KMeans( x, N_c ):
   #print x, N_c
   x = vq.whiten( x.T )
   x_est, idx = vq.kmeans2( x, N_c )
   return x_est.T

## TODO fix for large number of particles, speed-up
def KMeansPlusPlus( X, k ):
   L = np.array( [] )
   L1 = 0
   rv = uniform()

   while( np.unique( np.array( L ) ).size != k ):
      # initialization
      C = X[ :, np.round( rv.rvs( 1 ) * ( X.shape[ 1 ] - 1 ) ).astype( int ) ]
      L = np.zeros( X.shape[ 1 ] )
      
      for i in range( 1, k ):
         D = X-C[ :, L.astype( int ) ]
         #D = np.cumsum( np.sqrt( np.diag( np.dot( D.T, D ) ) ) )
         D = np.cumsum( np.sqrt( np.einsum('ij,ij->j', D, D) ) )
         if( D[ -1 ] == 0 ):
            C[ :, i:k+1 ] = X[ :, ones( 1, k+1-i ) ]
            print "EXIT!!"
            return ( C, L )
         C = np.matrix( np.c_[ ( C, X[ :, np.where( rv.rvs( 1 ) < D / D[ -1 ] )[0][0] ] ) ] )
         #C = np.c_[ ( C, X[ :, np.where( rv.rvs( 1 ) < D / D[ -1 ] )[0][0] ] ) ]
         #L = np.array( np.argmax( 2*np.real( np.dot( C.H, X ) ) - np.matrix( np.diag( np.dot( C.T, C ) ) ).T, axis=0 ) ) # don't use matrix mulitplication here!!!
         L = np.array( np.argmax( 2*np.real( np.dot( C.H, X ) ) - np.matrix( np.einsum('ij,ij->j', C, C) ).T, axis=0 ) ) # don't use matrix mulitplication here!!!
         
      while( np.any( L != L1 ) ):
         L1 = L
         for i in range( k ):
            l = (L == i).astype( int )
            l.shape = ( l.size, )
            C[ :, i ] = np.matrix( np.sum( X[:, np.array( np.where( l )[0] ) ], axis=1 ) / float( np.sum( l ) ) )
         #L = np.array( np.argmax( 2*np.real( np.dot( C.H, X ) ) - np.matrix( np.diag( np.dot( C.T, C ) ) ).T, axis=0 ) ) # don't use matrix mulitplication here!!!
         L = np.array( np.argmax( 2*np.real( np.dot( C.H, X ) ) - np.matrix( np.einsum('ij,ij->j', C, C) ).T, axis=0 ) ) # don't use matrix mulitplication here!!!
   L.shape = ( L.size, )   # unnecessary complexity, make L in the right shape from the beginning
   print "DONE", k
   return C, L

   
   
import random 
def cluster_points(X, mu):
   clusters  = {}
   for x in X:
      bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                 for i in enumerate(mu)], key=lambda t:t[1])[0]
      try:
         clusters[bestmukey].append(x)
      except KeyError:
         clusters[bestmukey] = [x]
   return clusters
 
def reevaluate_centers(mu, clusters):
   newmu = []
   keys = sorted(clusters.keys())
   for k in keys:
      newmu.append(np.mean(clusters[k], axis = 0))
   return newmu
 
def has_converged(mu, oldmu):
   return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def LLoyd(X, K):
   X = X.T.tolist()
   # Initialize to K random centers
   oldmu = random.sample(X, K)
   mu = random.sample(X, K)
   while not has_converged(mu, oldmu):
      oldmu = mu
      # Assign all points in X to clusters
      clusters = cluster_points(X, mu)
      # Reevaluate centers
      mu = reevaluate_centers(oldmu, clusters)
   return(mu, clusters)
   
#================================================================================
