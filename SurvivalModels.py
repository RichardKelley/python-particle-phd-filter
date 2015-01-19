import numpy as np

class UniformSurvival:
   def __init__( self, p_s ):
      # survival probability
      self.p_s = p_s
   
   def Evolute( self, x, w ):
      # uniform survival, position of particles is not considered
      return w * self.p_s
