-------------------------------------------------------------------------------------
               Python Particle Probability Hypothesis Density Filter
                         (python-particle-phd-filter)
-------------------------------------------------------------------------------------

This is a Python implementation of the Particle Probability Hypothesis density (PHD) 
filter described in:

[1] B.-N. Vo, S. Singh, and A. Doucet, “Sequential Monte Carlo implementation 
    of the PHD filter for multi-target tracking,” in Information Fusion, 2003. 
    Proceedings of the Sixth International Conference of, vol. 2, pp. 792 –799, Jul. 2003.

Required software packages are:
 - Python 2.7
 - numpy
 - scipy
 - matplotlib

'python-particle-phd-filter' was implemented as fun project. 
No guarantee shall be given regarding the correctness and completeness of 
the code presented. 


USAGE:
=====

Execute the python script "RunSimulation.py" from a console/terminal. This will simulate a 
randomly-generated scene of a varying number of moving targets being tracked in a 2D space.

RunSimulation.py makes use of the folowing modules:
- PPHDFilter.py
  the core module of the python particle PHD filter
- MeasurementModels
  definition of the measurement model in use for the simulation
- TransitionModels
  definition of the transition model in use for the simulation
- PriorModels
  definition of the prior model in use for the simulation
- BirthModels
  definition of the birth model in use for the simulation
- ClutterModels
  definition of the clutter model in use for the simulation
- SurvivalModels
  definition of the survival model in use for the simulation
- Systems
  definition of the system in use for the simulation
- ResamplingMethods
  definition of the resampling method in use for the simulation
- EstimationMethods
  definition of the estimation method in use for the simulation

LICENCE:
=======

(c) 2015 Rafael Karrer, Vienna.
All rights reserved.

    'python-particle-phd-filter' is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    'python-particle-phd-filter' is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gmphd.  If not, see <http://www.gnu.org/licenses/>.