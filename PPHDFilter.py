import numpy as np
import matplotlib.pyplot as plt

class PPHDFilter:
   def __init__( self ):
      self.particles = None
      self.weights = None
      self.T_est = 0
      self.measurements = None
      self.clutter = None
      
      self.TransitionModel = None
      self.MeasurementModel = None
      self.PriorModel = None
      self.BirthModel = None
      
   def RegisterTransitionModel( self, TransitionModel ):
      # transition model
      self.TransitionModel = TransitionModel
      # get dimension of state vector
      self.dimension = self.TransitionModel.GetDimension()

   def RegisterMeasurementModel( self, MeasurementModel ):
      # measurement model
      self.MeasurementModel = MeasurementModel

   def RegisterPriorModel( self, PriorModel ):
      # prior model
      self.PriorModel = PriorModel

   def RegisterBirthModel( self, BirthModel, N_b ):
      # number of new-born particles
      self.N_b = N_b
      # birth model
      self.BirthModel = BirthModel
      
   def RegisterSurvivalModel( self, SurvivalModel ):
      # survival model
      self.SurvivalModel = SurvivalModel

   def SetParticlesPerObject( self, N_t ):
      # number of particles per object
      # TODO: consider also operation with fixed number of particles
      self.N_t = N_t
      
   def RegisterClutterModel( self, ClutterModel ):
      # clutter model
      self.ClutterModel = ClutterModel
   
   def RegisterResamplingMethod( self, ResamplingMethod ):
      # resampling method
      self.Resampling = ResamplingMethod
   
   def RegisterEstimationMethod( self, EstimationMethod ):
      # estimation method
      self.Estimation = EstimationMethod
      
   def AssignMeasurements( self, measurements ):
      self.measurements = measurements
      
   def SurvivalModel( self, SurvivalModel ):
      # survival model
      sel.SurvivalModel = SurvivalModel
      
   def Initialize( self ):
      # sample particles from prior model
      self.particles = self.PriorModel.Sample( self.N_t )
      self.weights = self.PriorModel.Weight( self.N_t )
      
   def Predict( self ):
      # sample from transition model
      self.particles = self.TransitionModel.AdvanceState( self.particles )
      
      # Apply survival model to weights
      self.weights = self.SurvivalModel.Evolute( self.particles, self.weights )
      
      # sample from birth model
      new_particles = self.BirthModel.Sample( self.N_b )
      new_weights = self.BirthModel.Weight( self.N_b )
      
      print 'Birth, particle mass: ', np.sum( new_weights )
      
      # merge new and old particles
      old_particles = np.copy( self.particles )
      self.particles = np.zeros( old_particles.size + new_particles.size )
      self.particles = self.particles.reshape( self.dimension, self.T_est * self.N_t + self.N_b  )
      self.particles = np.matrix( self.particles )
      self.particles[ 0:, 0:self.T_est*self.N_t ] = old_particles
      self.particles[ 0:, self.T_est*self.N_t:self.T_est*self.N_t+self.N_b ] = new_particles
   
      # merge new and old weights
      old_weights = np.copy( self.weights )
      self.weights = np.zeros( old_weights.size + new_weights.size )
      self.weights[ 0:self.T_est*self.N_t ] = old_weights
      self.weights[ self.T_est*self.N_t:self.T_est*self.N_t+self.N_b ] = new_weights
      
      print 'Predict, particle mass: ', np.sum( self.weights )
      
   def Update( self, measurements ):
      # assign measurements (including clutter)
      self.measurements = measurements
      # prepare matrix for inner products
      num_of_measurements = self.measurements.shape[ 1 ]
      # TODO: define self.weights.size using variable
      stacked_likelihood = np.matrix( np.zeros( num_of_measurements * self.weights.size ).reshape( num_of_measurements, self.weights.size ) )
      for i in range( num_of_measurements ):
         stacked_likelihood[ i, : ] = self.MeasurementModel.CalcWeight( self.measurements[ :, i ], self.particles )
      # calculate inner product
      inner_products = stacked_likelihood * np.matrix( self.weights ).T
      #print 'inner_products: ', inner_products
      for i in range( num_of_measurements ):
         #print inner_products[ i, 0 ]
         stacked_likelihood[ i, : ] = stacked_likelihood[ i, : ] / ( inner_products[ i, 0 ] + self.ClutterModel.Likelihood( self.measurements[ :, i ] ) )

      # update weights
      #print "XXXXXXXXXXXXXXXXXXXXX", np.sum( np.array(sum( stacked_likelihood )) )
      #print 'Before Update, particle mass: ', np.sum( self.weights )
      #plt.plot( np.array(sum(stacked_likelihood)).T);plt.show();plt.close()
      #print 'Update, shapes: ', self.weights.shape, self.MeasurementModel.DetectionProbablitiy( self.particles ).shape, np.array( sum( stacked_likelihood ) ).shape, self.particles.shape
      self.weights = np.multiply( self.weights, np.ones( self.weights.size ) - self.MeasurementModel.DetectionProbablitiy( self.particles ) + np.array( sum( stacked_likelihood ) ) )  # don't use np.sum here!!
      #plt.plot( self.weights.T);plt.show();plt.close()
      # clear any 'NaN's
      self.weights[ np.where( np.isnan( self.weights ) ) ] = 0
      # clear any 'inf's
      self.weights[ np.where( np.isinf( self.weights ) ) ] = 0
      
      print 'Update, particle mass: ', np.sum( self.weights )

      
   def Resample( self ):
      particle_mass = np.sum( self.weights )
      # TODO: make T_est array
      self.T_est = int( np.round( particle_mass ) )
      indices = self.Resampling( self.weights / particle_mass, self.T_est * self.N_t )
      self.particles = self.particles[ :, indices ]
      #self.weights = np.ones( self.T_est * self.N_t ) * particle_mass / ( self.T_est * self.N_t )
      self.weights = np.ones( self.T_est * self.N_t ) / ( self.T_est * self.N_t )
      print 'Resample, particle mass: ', np.sum( self.weights )
      
   def Estimate( self ):
      if( self.particles.size != 0 ):
         self.x_est = self.Estimation( self.particles[0:2,:], self.T_est )
      else:
         self.x_est = np.matrix([])
         self.x_est.shape = ( self.dimension, 0 )
      return self.x_est