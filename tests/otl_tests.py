import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import CHASM as ch

from scipy.spatial.transform import Rotation as R

#add shower axis
zenith = np.radians(90.)
azimuth = 0
dca = 20000.

sim = ch.ShowerSimulation()
sim.add(ch.OverLimbAxis(zenith,azimuth,curved=True,ground_level=dca))

#add grid of detectors
n_side = 30
grid_width = 10000.

y = np.linspace(-grid_width, grid_width, n_side)
z = np.linspace(-grid_width, grid_width, n_side)
yy, zz = np.meshgrid(y,z)
xx = np.full_like(yy,-500000.) 

tel_vecs = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

sim.add(ch.SphericalCounters(tel_vecs, np.sqrt(1/np.pi)))

# add shower profile
t = np.load('test.npz')
X = t['slant_depth']
nch = t['charged_particles']
sim.add(ch.UserShower(X,nch))

# xmax = 736.
# nmax = 36002207.184834905
# x0 = 0
# Lambda = 52.75
# sim.add(ch.GHShower(xmax,nmax,x0,Lambda))

#add wavelength yield interval
sim.add(ch.Yield(270,1000,N_bins=1))

#run simulation
sig = sim.run(mesh=False, att=False)
photons = sig.photons.sum(axis=1)

#plot signal at each detector
fig = plt.figure()
cy = yy.flatten()*1.e-3
cz = zz.flatten()*1.e-3
h2d = plt.hist2d(cy,cz,weights=sig.photons.sum(axis=2).sum(axis=1),bins=n_side)
plt.suptitle('Cherenkov Over Limb Shower Signal at 525km Altitude')
# plt.title(f'Xmax = {xmax:.1f}, Nmax = {nmax:.1e}, X0 = {x0}, lambda = {Lambda}')
plt.xlabel('Counter Plane X-axis (km)')
plt.ylabel('Counter Plane Y-axis (km)')
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar(label = 'Number of Cherenkov Photons / m^2')

arc_angle = 2
tharc = np.linspace(-np.radians(arc_angle),3*np.radians(arc_angle),100)
pharc = np.linspace(0,2*np.pi,100)
eth, eph = np.meshgrid(tharc,pharc)
x_surf = sim.axis.earth_radius * np.sin(eth) * np.cos(eph)
y_surf = sim.axis.earth_radius * np.sin(eth) * np.sin(eph)
z_surf = sim.axis.earth_radius * np.cos(eth) - sim.axis.earth_radius - dca


photon_sum = photons.sum(axis=0)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(sig.source_points[:,0],sig.source_points[:,1],sig.source_points[:,2],c='r', s=100*photon_sum/photon_sum.max(), alpha=0.2)
ax.scatter(xx,yy,zz,c='k',label='counters')
ax.plot_surface(x_surf,y_surf,z_surf)
ax.set_aspect('equal')
# ax.set_zlim(-1.e5,1.e5)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
