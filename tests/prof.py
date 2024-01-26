import cProfile
import numpy as np
from scipy.spatial.transform import Rotation as R

import CHASM as ch

#add shower axis
zenith = np.radians(85)
azimuth = 0.
sim = ch.ShowerSimulation()
sim.add(ch.UpwardAxis(zenith,azimuth,curved=True))

#add grid of detectors
n_side = 30
grid_width = 100000.
detector_grid_alt = 525. #km

x = np.linspace(-grid_width, grid_width, n_side)
y = np.linspace(-grid_width, grid_width, n_side)
xx, yy = np.meshgrid(x,y)
r = sim.ingredients['axis'].h_to_axis_R_LOC(detector_grid_alt*1.e3,zenith) #get distance along axis corresponding to detector altitude
zz = np.full_like(xx, r) #convert altitude to m

vecs = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

theta_rot_axis = np.array([0,1,0])
theta_rotation = R.from_rotvec(theta_rot_axis * zenith)

z_rot_axis = np.array([0,0,1])
z_rotation = R.from_rotvec(z_rot_axis * np.pi/2)

vecs = z_rotation.apply(vecs)
tel_vecs = theta_rotation.apply(vecs)
sim.add(ch.SphericalCounters(tel_vecs, np.sqrt(1/np.pi)))

#add shower profile
t = np.load('test.npz')
X = t['slant_depth']
nch = t['charged_particles']
sim.add(ch.UserShower(X,nch))

#add wavelength yield interval
sim.add(ch.Yield(270,1000,N_bins=1))

#run simulation and profile
cProfile.run('sim.run(mesh=True, att=True)')