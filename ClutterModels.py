import numpy as np
from scipy.stats import poisson
from scipy.stats import uniform

class PoissonClutter:
   def __init__( self, system, lambda_c ):
      self.lambda_c = lambda_c
      self.system = system
      self.dim = self.system.GetDimensions()
      self.rc = poisson( lambda_c )
      self.rb = uniform()
      
   def Likelihood( self, z ):
      # uniformly distributed clutter
      #return 1.0 * self.lambda_c
      return 1.0 * self.lambda_c / self.system.GetVolume() * 900 # this is needed because sensor only works in 2 dimensions
      
   def Sample( self ):
      if( self.lambda_c > 0 ):
         n = self.rc.rvs( 1 )
         clutter = np.matrix( [ ] )
         clutter.shape = ( 0, n )
         for i in range( self.dim.shape[ 0 ] ):
            clutter = np.vstack( ( clutter, self.rb.rvs( n ) * ( self.dim[ i, 1 ] - self.dim[ i, 0 ] ) + self.dim[ i, 0 ] ) )
      else:
         clutter = np.array( [] )
         clutter.shape = ( self.dim.shape[0], 0 )
      return np.matrix( clutter )      
      