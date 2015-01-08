import numpy as np
from scipy.stats import multivariate_normal as mv_normal

class XYRandomWalk:
   def __init__( self, var_v, p_s ):
      self.A = np.matrix([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
      self.Q = var_v * np.matrix([ [ 1.0/3, 0    ,1.0/2,0     ]
                    ,[ 0    , 1.0/3, 0   ,1.0/2 ]
                    ,[ 1.0/2, 0    , 1   ,0     ]
                    ,[ 0    , 1.0/2, 0   ,1     ] ])
      self.rv = mv_normal( np.array([0,0,0,0]), self.Q)
      
      self.p_s = p_s
      
   def AdvanceState( self, x ):
      v = np.matrix( self.rv.rvs( 1 ) ).T
      return self.A * x + v
      
   def GetDimension( self ):
      return len( self.A )
      #return 4

      