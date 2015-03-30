import numpy as np
from scipy.stats import uniform

# FIXME: this is not Poisson birth! This is Un
class PoissonBirth:
   def __init__( self, system, lambda_b ):
      self.lambda_b = lambda_b
      self.system = system
      self.rb = uniform()
      
   def Sample( self, n ):
      dim = self.system.GetDimensions()
      
      particles = np.array( [ ] )
      # FIXME: this is probably slow?
      for i in range( dim.shape[ 0 ] ):
         particles = np.append( particles, self.rb.rvs( n ) * ( dim[ i, 1 ] - dim[ i, 0 ] ) + dim[ i, 0 ] )

      particles = particles.reshape( dim.shape[ 0 ], n )
      return np.matrix( particles )

   def Weight( self, n ):
      print self.lambda_b
      return  np.ones( n ) * 1.0 / n * self.lambda_b