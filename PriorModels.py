import numpy as np
from scipy.stats import uniform


class UniformPrior:
   def __init__( self, dimensions ):
      self.rb = uniform()
      self.dim = dimensions
      
      
   def Sample( self, n ):
      particles = np.array( [ ] )
      for i in range( self.dim.shape[ 0 ] ):
         particles = np.append( particles, self.rb.rvs( n ) * ( self.dim[ i, 1 ] - self.dim[ i, 0 ] ) + self.dim[ i, 0 ] )

      particles.reshape( self.dim.shape[ 0 ], n )
      return np.matrix( particles )
   
   def Weight( self, n ):
      return  np.matrix( np.ones( 4 * n ).reshape( 4, n ) ) * 1.0 / n
       
class NoPrior:
   def __init__( self ):
      return None
      
   def Sample( self, n ):
      return np.matrix( [ [], [], [], []  ] )
   
   def Weight( self, n ):
      return  np.matrix( [ [] ] )