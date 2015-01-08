import numpy as np
from scipy.stats import multivariate_normal as mv_normal
from scipy.stats import uniform

class XYSensorWithAWGN:
   def __init__( self, system, var_z, p_d ):
      self.u_z = np.array( [ 0, 0 ] )
      self.C_z = var_z * np.identity( 2 )
      self.r_z = mv_normal( self.u_z, self.C_z )
      self.r_d = uniform()
      self.system = system
      
      #detection probability
      self.p_d = p_d

   def Likelihood( self, z ):
      f_z = self.r_z.pdf( z ) 
      return f_z
      
   def Sample( self, n ):
      return self.r_z.rvs( n )
      
   def CalcWeight( self, z, x ):
      w_x = self.DetectionProbablitiy( x ) * self.Likelihood( ( z[ 0:2, : ] - x[ 0:2, : ] ).T )
      return w_x
   
   def DetectionProbablitiy( self, x ):
      oob = np.logical_not( self.system.IsOutOfBounds( x ) )
      # detection probability != 1 increases particle weight over time!!! ( due to 1 - p_d + sum( ... ) )
      return np.array( [ float(j) for j in oob ] ) * self.p_d
      #return 1

   def Measure( self, x ):
      n = self.Sample( x.shape[1] * 2 ).T.reshape( 4, x.shape[1] )
      z = x + n
      det = np.where( self.r_d.rvs( x.shape[1] ) <= self.p_d )[0]
      z[ 2:4, : ] = 0
      return z[ :, det ]