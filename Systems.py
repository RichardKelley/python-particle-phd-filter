import numpy as np

class BoxWorld4D:
   def __init__( self, x0dim, x1dim, x2dim, x3dim ):
      self.x0dim = x0dim
      self.x1dim = x1dim
      self.x2dim = x2dim
      self.x3dim = x3dim
      
      self.x0size = self.x0dim[1] - self.x0dim[0]
      self.x1size = self.x1dim[1] - self.x1dim[0]
      self.x2size = self.x2dim[1] - self.x2dim[0]
      self.x3size = self.x3dim[1] - self.x3dim[0]
   
   def GetDimensions( self ):
      return np.matrix( [ self.x0dim, self.x1dim, self.x2dim, self.x3dim ] )
      
   def GetVolume( self ):
      #return 1.0
      return 1.0 * self.x0size * self.x1size * self.x2size * self.x3size
      
   def IsOutOfBounds( self, x ):
      oob0 = np.array( [ x[ 0, j ] < self.x0dim[ 0 ] or  x[ 0, j ] > self.x0dim[ 1 ] for j in range( x.shape[ 1 ] ) ] )
      oob1 = np.array( [ x[ 1, j ] < self.x1dim[ 0 ] or  x[ 1, j ] > self.x1dim[ 1 ] for j in range( x.shape[ 1 ] ) ] )
      oob2 = np.array( [ x[ 2, j ] < self.x2dim[ 0 ] or  x[ 2, j ] > self.x2dim[ 1 ] for j in range( x.shape[ 1 ] ) ] )
      oob3 = np.array( [ x[ 3, j ] < self.x3dim[ 0 ] or  x[ 3, j ] > self.x3dim[ 1 ] for j in range( x.shape[ 1 ] ) ] )
      oob = oob0 + oob1 + oob2 + oob3
      return np.array( [ j > 0 for j in oob ] )