import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

# TODO: is this really systematic resampling?
def SystematicResampling( weights, N_s ):
   if N_s < weights.size:
      N = weights.size
   else:
      N = N_s

   rv = uniform()
   indices = np.arange( N_s, dtype=np.int32 )
   # cumulative sum of weights
   Q = 1.0 * np.cumsum( weights ) / np.sum( weights )
   T = 1.0 * np.arange( N + 1 ) / N 
   print T, Q
   i = 0
   j = 0
   
   if( np.sum( weights ) != 0 and N_s != 0 ):
      while( i < N ):
         if( T[ i ] < Q[ j ] ):
            indices[ i ] = j
            i = i + 1
         else:
            j = j + 1
   
   randint = np.array( np.round( rv.rvs( N_s ) * ( N_s - 1 ) ), dtype=np.int32 )
   print indices
   indices = indices[ randint ]
   
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
   
   
       # N = length(w);
    # Q = cumsum(w);

    # T = linspace(0,sum(w)-1/N,N) + rand(1)/N;
    # T(N+1) = 1;

    # i=1;
    # j=1;

    # indx = linspace(1,L,L);
   
    # if (sum(w) ~= 0)
        # while (i<=N),
            # if (T(i)<Q(j)),
                # indx(i)=j;
                # i=i+1;
            # else
                # j=j+1;        
            # end
        # end
    # end
    # indx = indx(round((N-1)*rand(L,1))+1);
# end