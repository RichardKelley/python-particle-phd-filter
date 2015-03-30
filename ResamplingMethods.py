import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

# TODO: is this really systematic resampling?
def SystematicResampling( weights, N_s ):
   N = weights.size
   #if N_s < weights.size:
   #   N = weights.size
   #else:
   #   N = N_s

   rv = uniform()

   # cumulative sum of weights
   Q = 1.0 * np.cumsum( weights )
   T = np.zeros( N + 1 )
   T[ 0:N ] = 1.0 * np.arange( N ) * ( np.sum( weights ) - 1.0 / N ) / ( N - 1 ) + 1.0 * rv.rvs( 1 ) / N
   T[ -1 ] = 1
   
   indices = np.arange( N, dtype=np.int32 )

   #plt.plot( Q );plt.plot( T ); plt.show(); plt.close()
   
   i = 0
   j = 0
   
   if( np.sum( weights ) != 0 ):
      while( i < N ):
         #print i, j, T[i], Q[j], N
         if( T[ i ] < Q[ j ] ):
            indices[ i ] = j
            i = i + 1
         else:
            j = j + 1
   
   randint = np.array( np.round( rv.rvs( N_s ) * ( N - 1 ) ), dtype=np.int32 )
   indices = indices[ randint ]
   #indices = indices - 1
   
   #print n, N_s
   #plt.plot( Q );plt.plot( T ); plt.show(); plt.close()
   
   return indices


def RafaelsMethod( weights, N_s ):
   rv = uniform()
   # TODO: sort weights for quicker search
   # cumulative sum of weights
   csum = 1.0 * np.cumsum( weights ) / np.sum( weights )
   rand = rv.rvs( N_s )
   indices = np.zeros( N_s, dtype=np.int32 )
   
   for i in range( N_s ):    
      indices[ i ] = np.where( rand[ i ] < csum )[0][0] 
      
   return indices