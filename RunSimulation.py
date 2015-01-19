import numpy as np
from scipy.stats import multivariate_normal as mv_normal
from scipy.stats import uniform
import matplotlib.pyplot as plt

import PPHDFilter

import MeasurementModels
import TransitionModels
import PriorModels
import BirthModels
import ClutterModels
import Systems
import SurvivalModels
import ResamplingMethods
import EstimationMethods

O = 5
N = 50                                         #number of iterations
NO = np.zeros( N+1 )                           # initial number of objects
lambda_b = 1.0 / N * O                # O objects are born over all iterations

system = Systems.BoxWorld4D( np.array([0,100]), np.array([0,100]), np.array([-15,15]), np.array([-15,15]) )
priorModel = PriorModels.NoPrior()
birthModel = BirthModels.PoissonBirth( system, lambda_b )
survivalModel = SurvivalModels.UniformSurvival( p_s = 1.0 )
clutterModel = ClutterModels.PoissonClutter( system, lambda_c = 0.0 )
sensor = MeasurementModels.XYSensorWithAWGN( system, var_z = 1.0, p_d = 0.98 )

rb = uniform()

#initialize empty 
allObj = np.array([], dtype=object)
allMeas = np.array([], dtype=object)

process = TransitionModels.XYRandomWalk( 1, 1 )

for n in range( 0, N ):
   meas = np.matrix( [] )
   meas.shape = ( process.GetDimension(), 0 )
   
   for no in range(int(NO[n])):
      allObj[no][:,n] = process.AdvanceState( allObj[no][:,n-1] )
      meas = np.append( meas, sensor.Measure( allObj[no][:,n] ), axis=1 )
      
   if( rb.rvs() < lambda_b ):
      NO[n] = NO[n] + 1
      x0 = birthModel.Sample( 1 )
      x0[2, 0] = 0
      x0[3, 0] = 0
      allObj = np.append( allObj, [0] )
      ## TODO: make matrix start at index n only, doesn't need to hold values for all n
      allObj[-1] = np.matrix( np.zeros(N * 4).reshape(4, N) )
      allObj[-1][:,n] = x0
      meas = np.append( meas, sensor.Measure( allObj[-1][:,n] ), axis=1 )
   
   meas = np.append( meas, clutterModel.Sample(), axis=1 )

   allMeas = np.append( allMeas, [0] )
   allMeas[-1] = meas
   NO[n + 1] = NO[n]
   #print n, NO[n]

pphdFilter = PPHDFilter.PPHDFilter()
pphdFilter.SetParticlesPerObject( 500 )
pphdFilter.RegisterTransitionModel( process )
pphdFilter.RegisterMeasurementModel( sensor )
pphdFilter.RegisterPriorModel( priorModel )
pphdFilter.RegisterBirthModel( birthModel, 2000 )
pphdFilter.RegisterSurvivalModel( survivalModel )
pphdFilter.RegisterClutterModel( clutterModel )
pphdFilter.RegisterResamplingMethod( ResamplingMethods.SystematicResampling )
pphdFilter.RegisterEstimationMethod( EstimationMethods.KMeans )
#pphdFilter.RegisterEstimationMethod( EstimationMethods.KMeansPlusPlus )

pphdFilter.Initialize()

for n in range( 0, N ):
   # Prediction Step
   pphdFilter.Predict()
   
   # Update Step
   pphdFilter.Update( allMeas[ n ] )
   
   # Resample Step
   pphdFilter.Resample()
   
   # Estimation Step
   #x_est = pphdFilter.Estimate()
   
   # Visualize Data
   plt.plot( np.array( pphdFilter.particles[ 0, : ] ).T, np.array( pphdFilter.particles[ 1, : ] ).T, '.' );
   plt.plot( np.array( pphdFilter.measurements[ 0, :] ).T, np.array( pphdFilter.measurements[ 1, : ].T ), 'o' )
   plt.plot( np.array( allObj[0][0,:] ).T, np.array( allObj[0][1,:] ).T )
   plt.plot( np.array( allObj[0][0,n] ).T, np.array( allObj[0][1,n] ).T, 'o' )
   #plt.plot( np.array( x_est[0,:] ).T, np.array( x_est[1,:] ).T, 'x' )
   plt.xlim( system.GetDimensions()[0,0] , system.GetDimensions()[0,1] )
   plt.ylim( system.GetDimensions()[0,0] , system.GetDimensions()[0,1] )
   plt.show(); plt.close()