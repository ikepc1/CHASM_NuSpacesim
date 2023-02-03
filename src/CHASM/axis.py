import numpy as np
from abc import ABC, abstractmethod
from importlib.resources import as_file, files
from scipy.constants import value,nano
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm

from .atmosphere import *
from .generate_Cherenkov import MakeYield
from .shower import Shower

class Axis(ABC):
    '''This is the abstract base class which contains the methods for computing
    the cartesian vectors and corresponding slant depths of an air shower'''

    earth_radius = 6.371e6 #meters
    lX = -100. #This is the default value for the distance to the axis in log moliere units (in this case log(-inf) = 0, or on the axis)
    atm = USStandardAtmosphere()

    def __init__(self, zenith: float, azimuth: float, ground_level: float = 0.):
        self.zenith = zenith
        self.azimuth = azimuth
        self.ground_level = ground_level

    @property
    def zenith(self):
        '''polar angle  property getter'''
        return self._zenith

    @zenith.setter
    def zenith(self, zenith):
        '''zenith angle property setter'''
        if zenith >= np.pi/2:
            raise ValueError('Zenith angle cannot be greater than pi / 2')
        if zenith < 0.:
            raise ValueError('Zenith angle cannot be less than 0')
        self._zenith = zenith

    @property
    def azimuth(self):
        '''azimuthal angle property getter'''
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        '''azimuthal angle property setter'''
        if azimuth >= 2 * np.pi:
            raise ValueError('Azimuthal angle must be less than 2 * pi')
        if azimuth < 0.:
            raise Valueunattenuated_30degree_sealevelError('Azimuthal angle cannot be less than 0')
        self._azimuth = azimuth

    @property
    def ground_level(self):
        '''ground level property getter'''
        return self._ground_level

    @ground_level.setter
    def ground_level(self, value):
        '''ground level property setter'''
        if value > self.atm.maximum_height:
            raise ValueError('Ground level too high')
        self._ground_level = value

    @property
    def altitude(self):
        '''altitude property definition'''
        return np.linspace(self.ground_level, self.atm.maximum_height, 1000)

    @property
    def dh(self):
        '''This method sets the dh attribute'''
        dh = self.h[1:] - self.h[:-1]
        return np.concatenate((np.array([0]),dh))

    @property
    def h(self):
        '''This is the height above the ground attribute'''
        hs = self.altitude - self.ground_level
        hs[0] = 1.e-5
        return hs

    @property
    def delta(self):
        '''delta property definition (index of refraction - 1)'''
        return self.atm.delta(self.altitude)

    @property
    def density(self):
        '''Axis density property definition (kg/m^3)'''
        return self.atm.density(self.altitude)

    @property
    def moliere_radius(self):
        '''Moliere radius property definition (m)'''
        return 96. / self.density

    def h_to_axis_R_LOC(self,h,theta):
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights (m above ground level)
        theta: polar angle of shower axis (radians)

        returns: r (m) (same size as h), an array of distances along the shower
        axis_sp.
        '''
        cos_EM = np.cos(np.pi-theta)
        R = self.earth_radius + self.ground_level
        r_CoE= h + R # distance from the center of the earth to the specified height
        rs = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
        # rs -= rs[0]
        # rs[0] = 1.
        return rs # Need to find a better way to define axis zero point, currently they are all shifted by a meter to prevent divide by zero errors

    @classmethod
    def theta_normal(cls,h,r):
        '''This method calculates the angle the axis makes with respect to
        vertical in the atmosphere (at that height).

        Parameters:
        h: array of heights (m above sea level)
        theta: array of polar angle of shower axis (radians)

        Returns:
        The corrected angles(s)
        '''
        cq = (r**2 + h**2 + 2*cls.earth_radius*h)/(2*r*(cls.earth_radius+h))
        cq[cq>1.] = 1.
        cq[cq<-1.] = -1.
        # cq = ((cls.earth_radius+h)**2+r**2-cls.earth_radius**2)/(2*r*(cls.earth_radius+h))
        return np.arccos(cq)

    @property
    def theta_difference(self) -> np.ndarray:
        '''This property is the difference between the angle a vector makes with
        the z axis and the angle it makes with vertical in the atmosphere at
        all the axis heights.
        '''
        return self.zenith - self.theta_normal(self.h, self.r)

    @property
    def dr(self):
        '''This method sets the dr attribute'''
        dr = self.r[1:] - self.r[:-1]
        return np.concatenate((np.array([0]),dr))

    @property
    def vectors(self):
        '''axis vector property definition

        returns a vector from the origin to a distances r
        along the axis'''
        ct = np.cos(self.zenith)
        st = np.sin(self.zenith)
        cp = np.cos(self.azimuth)
        sp = np.sin(self.azimuth)
        axis_vectors = np.empty([np.shape(self.r)[0],3])
        axis_vectors[:,0] = self.r * st * cp
        axis_vectors[:,1] = self.r * st * sp
        axis_vectors[:,2] = self.r * ct
        return axis_vectors

    def depth(self, r: np.ndarray):
        '''This method is the depth as a function of distance along the shower
        axis'''
        return np.interp(r, self.r, self.X)

    @property
    @abstractmethod
    def r(self):
        '''r property definition'''
        # return self.h_to_axis_R_LOC(self.h, self.zenith)

    @property
    @abstractmethod
    def X(self):
        '''This method sets the depth along the shower axis attribute'''

    @abstractmethod
    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''

    @abstractmethod
    def theta(self):
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''

    @abstractmethod
    def get_timing(self):
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_attenuation(self):
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_gg_file(self):
        '''This method should return the gg array for the particular axis type.
        For linear axes, it should return the regular gg array. For a mesh axis,
        it should return the gg array for the axis' particular log(moliere)
        interval.
        '''

def vector_magnitude(vectors: np.ndarray):
    '''This method computes the length of an array of vectors'''
    return np.sqrt((vectors**2).sum(axis = -1))

class Counters(ABC):
    '''This is the class containing the neccessary methods for finding the
    vectors from a shower axis to a user defined array of Cherenkov detectors
    with user defined size'''

    def __init__(self, input_vectors: np.ndarray, input_radius: np.ndarray):
        self.vectors = input_vectors
        self.input_radius = input_radius

    @property
    def vectors(self):
        '''Vectors to user defined Cherenkov counters getter'''
        return self._vectors

    @vectors.setter
    def vectors(self, input_vectors: np.ndarray):
        '''Vectors to user defined Cherenkov counters setter'''
        if type(input_vectors) != np.ndarray:
            input_vectors = np.array(input_vectors)
        if input_vectors.shape[1] != 3 or len(input_vectors.shape) != 2:
            raise ValueError("Input is not an array of vectors.")
        self._vectors = input_vectors

    @property
    def input_radius(self):
        '''This is the input counter radius getter.'''
        return self._input_radius

    @input_radius.setter
    def input_radius(self, input_value):
        '''This is the input counter radius setter.'''
        if type(input_value) != np.ndarray:
            input_value = np.array(input_value)
        if np.size(input_value) == np.shape(self.vectors)[0] or np.size(input_value) == 1:
            self._input_radius = input_value
        else:
            raise ValueError('Counter radii must either be a single value for all detectors, or a list with a radius corresponding to each defined counter location.')

    @property
    def N_counters(self):
        '''Number of counters.'''
        return self.vectors.shape[0]

    @property
    def r(self):
        '''distance to each counter property definition'''
        return vector_magnitude(self.vectors)

    @abstractmethod
    def area(self, *args, **kwargs):
        '''This is the abstract method for the detection surface area normal to
        the axis as seen from each point on the axis. Its shape must be
        broadcastable to the size of the travel_length array, i.e. either a
        single value or (# of counters, # of axis points)'''

    @abstractmethod
    def omega(self, *args, **kwargs):
        '''This abstract method should compute the solid angle of each counter
        as seen by each point on the axis'''

    def area_normal(self):
        '''This method returns the full area of the counting aperture.'''
        return np.pi * self.input_radius**2

    def law_of_cosines(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        '''This method returns the angle C across from side c in triangle abc'''
        cos_C = (c**2 - a**2 - b**2)/(-2*a*b)
        cos_C[cos_C > 1.] = 1.
        cos_C[cos_C < -1.] = -1.
        return np.arccos(cos_C)

    def travel_vectors(self, axis_vectors):
        '''This method returns the vectors from each entry in vectors to
        the user defined array of counters'''
        return self.vectors.reshape(-1,1,3) - axis_vectors

    def travel_length(self, axis_vectors):
        '''This method computes the distance from each point on the axis to
        each counter'''
        return vector_magnitude(self.travel_vectors(axis_vectors))

    def cos_Q(self, axis_vectors):
        '''This method returns the cosine of the angle between the z-axis and
        the vector from the axis to the counter'''
        travel_n = self.travel_vectors(axis_vectors) / self.travel_length(axis_vectors)[:,:,np.newaxis]
        return np.abs(travel_n[:,:,-1])

    def travel_n(self, axis_vectors) -> np.ndarray:
        '''This method returns the unit vectors pointing along each travel
        vector.
        '''
        return self.travel_vectors(axis_vectors) / self.travel_length(axis_vectors)[:,:,np.newaxis]

    def calculate_theta(self, axis_vectors):
        '''This method calculates the angle between the axis and counters'''
        travel_length = self.travel_length(axis_vectors)
        axis_length = np.broadcast_to(vector_magnitude(axis_vectors), travel_length.shape)
        counter_length = np.broadcast_to(self.r, travel_length.T.shape).T
        return self.law_of_cosines(axis_length, travel_length, counter_length)

class MakeSphericalCounters(Counters):
    '''This is the implementation of the Counters abstract base class for
    CORSIKA IACT style spherical detection volumes.

    Parameters:
    input_vectors: rank 2 numpy array of shape (# of counters, 3) which is a
    list of vectors (meters).
    input_radius: Either a single value for an array of values corresponding to
    the spherical radii of the detection volumes (meters).
    '''

    def __repr__(self):
        return "SphericalCounters({:.2f} Counters with average area~ {:.2f})".format(
        self.vectors.shape[0], self.area_normal().mean())

    def area(self):
        '''This is the implementation of the area method, which calculates the
        area of spherical counters as seen from the axis.
        '''
        return self.area_normal()

    def omega(self, axis: Axis):
        '''This method computes the solid angle of each counter as seen by
        each point on the axis'''
        return (self.area() / (self.travel_length(axis).T)**2).T

class MakeFlatCounters(Counters):
    '''This is the implementation of the Counters abstract base class for flat,
    horizontal counting apertures.

    Parameters:
    input_vectors: rank 2 numpy array of shape (# of counters, 3) which is a
    list of vectors (meters).
    input_radius: Either a single value for an array of values corresponding to
    the radii of the detection aperture (meters).'''

    def __repr__(self):
        return "FlatCounters({:.2f} Counters with average area~ {:.2f})".format(
        self.vectors.shape[0], self.area_normal().mean())

    def area(self, axis_vectors):
        '''This is the implementation of the area method for flat counting
        apertures. This method returns the area represented by each aperture as
        seen by each point on the axis. The returned array is of size
        (# of counters, # of axis points).'''
        return (self.cos_Q(axis_vectors).T * self.area_normal()).T

    def omega(self, axis_vectors):
        '''This method computes the solid angle of each counter as seen by
        each point on the axis
        (# of counters, # of axis points)'''
        return self.area(axis_vectors) / (self.travel_length(axis_vectors)**2)

class LateralSpread:
    '''This class interacts with the table of NKG universal lateral distributions
    '''
    with as_file(files('CHASM.data')/'n_t_lX_of_t_lX.npz') as file:
        lx_table = np.load(file)

    t = lx_table['ts']
    lX = lx_table['lXs']
    n_t_lX_of_t_lX = lx_table['n_t_lX_of_t_lX']

    @classmethod
    def get_t_indices(cls, input_t: np.ndarray):
        '''This method returns the indices of the stages in the table closest to
        the input stages.
        '''
        return np.abs(input_t[:, np.newaxis] - cls.t).argmin(axis=1)

    @classmethod
    def get_lX_index(cls, input_lX: float):
        '''This method returns the index closest to the input lX within the 5
        tabulated lX values.
        '''
        return np.abs(input_lX - cls.lX).argmin()

    @classmethod
    def nch_fractions(cls, input_ts: np.ndarray, input_lX: float):
        '''This method returns the fraction of charged particles at distance
        exp(lX) moliere units from the shower axis at an array of stages
        (input_ts).
        '''
        t_indices = cls.get_t_indices(input_ts)
        lX_index = cls.get_lX_index(input_lX)
        return cls.n_t_lX_of_t_lX[t_indices,lX_index]

def axis_to_mesh(lX: float, axis: Axis, shower: Shower, N_ring: int = 20) -> tuple:
    '''This function takes an shower axis and creates a 3d mesh of points around
    the axis (in coordinates where the axis is the z-axis)
    Parameters:
    axis: axis type object
    shower: shower type object
    Returns:
    an array of vectors to points in the mesh
    size (, 3)
    The corresponding # of charged particles at each point
    The corresponding array of stages (for use in universality calcs)
    The corresponding array of deltas (for use in universality calcs)
    The corresponding array of shower steps (dr) in m (for Cherenkov yield calcs)
    The corresponding array of altitudes (for timing calcs)

    '''
    X = np.exp(lX) #number of moliere units for the radius of the ring
    X_to_m = X * axis.moliere_radius
    axis_t = shower.stage(axis.X)
    total_nch = shower.profile(axis.X) * LateralSpread.nch_fractions(axis_t,lX)
    axis_d = axis.delta
    axis_dr = axis.dr
    axis_altitude = axis.altitude
    r = axis.r
    ring_theta = np.arange(0,N_ring) * 2 * np.pi / N_ring
    ring_x = X_to_m[:, np.newaxis] * np.cos(ring_theta)
    ring_y = X_to_m[:, np.newaxis] * np.sin(ring_theta)
    x = ring_x.flatten()
    y = ring_y.flatten()
    z = np.repeat(r, N_ring)
    t = np.repeat(axis_t, N_ring)
    d = np.repeat(axis_d, N_ring)
    nch = np.repeat(total_nch / N_ring, N_ring)
    dr = np.repeat(axis_dr, N_ring)
    a = np.repeat(axis_altitude, N_ring)
    return np.array((x,y,z)).T, nch, t, d, dr, a

def rotate_mesh(mesh: np.ndarray, theta: float, phi: float) -> np.ndarray:
    '''This function rotates an array of vectors by polar angle theta and
    azimuthal angle phi.

    Parameters:
    mesh: numpy array of axis vectors shape = (# of vectors, 3)
    theta: axis polar angle (radians)
    phi: axis azimuthal angle (radians)
    Returns a numpy array the same shape as the original list of vectors
    '''
    theta_rot_axis = np.array([0,1,0]) #y axis
    phi_rot_axis = np.array([0,0,1]) #z axis
    theta_rot_vector = theta * theta_rot_axis
    phi_rot_vector = phi * phi_rot_axis
    theta_rotation = R.from_rotvec(theta_rot_vector)
    phi_rotation = R.from_rotvec(phi_rot_vector)
    mesh_rot_by_theta = theta_rotation.apply(mesh)
    mesh_rot_by_theta_then_phi = phi_rotation.apply(mesh_rot_by_theta)
    return mesh_rot_by_theta_then_phi

class MeshAxis(Axis):
    '''This class is the implementation of an axis where the sampled points are
    spread into a mesh.
    '''
    lXs = np.arange(-6,0)

    def __init__(self, lX_interval: tuple, linear_axis: Axis, shower: Shower):
        self.lX_interval = lX_interval
        self.lX = np.mean(lX_interval)
        self.linear_axis = linear_axis
        self.shower = shower
        self.zenith = linear_axis.zenith
        self.azimuth = linear_axis.azimuth
        self.ground_level = linear_axis.ground_level
        mesh, self.nch, self._t, self._d, self._dr, self._a  = axis_to_mesh(self.lX, self.linear_axis, self.shower)
        self.rotated_mesh = rotate_mesh(mesh, linear_axis.zenith, linear_axis.azimuth)

    @property
    def lX_inteval(self):
        '''lX_inteval property getter'''
        return self._lX_inteval

    @lX_inteval.setter
    def lX_inteval(self, interval):
        '''lX_inteval angle property setter'''
        # if type(interval) != tuple:
        #     raise ValueError('lX interval needs to be a tuple')
        # if interval not in zip(self.lXs[:-1], self.lXs[1:]):
        #     raise ValueError('lX interval not in tabulated ranges.')
        self._lX_inteval = interval

    @property
    def delta(self):
        '''Override of the delta property so each one corresponds to its
        respective mesh point.'''
        return self._d

    @property
    def vectors(self):
        '''axis vector property definition

        returns vectors from the origin to mesh axis points.
        '''
        return self.rotated_mesh

    @property
    def r(self):
        '''overrided r property definition'''
        return vector_magnitude(self.rotated_mesh)

    @property
    def dr(self):
        '''overrided dr property definition'''
        return self._dr

    @property
    def altitude(self):
        '''overrided h property definition'''
        return self._a

    @property
    def X(self):
        '''This method sets the depth along the shower axis attribute'''
        return self.linear_axis.X

    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''
        return self.axis.distance

    def theta(self, axis_vectors, counters: Counters):
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''
        return self.linear_axis.theta(axis_vectors, counters)

    def get_timing(self, axis: Axis, counters: Counters):
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''
        return self.linear_axis.get_timing(axis, counters)

    def get_attenuation(self, axis: Axis, counters: Counters, y: MakeYield):
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''
        return self.linear_axis.get_attenuation(axis, counters, y)

    def get_gg_file(self) -> str:
        '''This method returns the gg array file for the axis' particular
        log(moliere) interval.
        '''
        start, end = self.find_nearest_interval()
        return f'gg_t_delta_theta_lX_{start}_to_{end}.npz'

    def find_nearest_interval(self) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.lXs[:-1],self.lX)
        if index == 0:
            return self.lXs[0], self.lXs[1]
        else:
            return self.lXs[index-1], self.lXs[index]

    # def get_gg_file(self):
    #     '''This method returns the original gg array file.
    #     '''
    #     # return 'gg_t_delta_theta_lX_-6_to_-5.npz'
    #     # return 'gg_t_delta_theta_mc.npz'
    #     return 'gg_t_delta_theta_2020_normalized.npz'

class MeshShower(Shower):
    '''This class is the implementation of a shower where the shower particles are
    distributed to a mesh axis rather than just the longitudinal axis.
    '''

    def __init__(self, mesh_axis: MeshAxis):
        self.mesh_axis = mesh_axis

    def stage(self, X: np.ndarray):
        '''This method returns the corresponding stage of each mesh point.
        '''
        return self.mesh_axis._t

    def profile(self, X: np.ndarray):
        '''This method returns the number of charged particles at each mesh
        point
        '''
        return self.mesh_axis.nch

class MakeUpwardAxis(Axis):
    '''This is the implementation of an axis for an upward going shower, depths
    are added along the axis in the upward direction'''

    @property
    def X(self):
        '''This method sets the depth attribute'''
        rho = self.atm.density(self.altitude)
        axis_deltaX = np.sqrt(rho[1:] * rho[:-1]) * self.dr[1:] / 10# converting to g/cm^2
        return np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))

    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X, self.r)

    def theta(self, axis_vectors, counters: Counters):
        '''In this case we need pi minus the interal angle across from the
        distance to the counter'''
        return np.pi - counters.calculate_theta(axis_vectors)

class MakeUpwardAxisFlatPlanarAtm(MakeUpwardAxis):
    '''This is the implementation of an upward going shower axis with a flat
    planar atmosphere'''

    def __repr__(self):
        return "UpwardAxisFlatPlanarAtm(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
                                        self.zenith, self.azimuth, self.ground_level)

    @property
    def r(self):
        '''This is the axis distance property definition'''
        return self.h / np.cos(self.zenith)

    def get_timing(self, axis: Axis, counters: Counters):
        '''This method returns the instantiated upward flat atm timing object'''
        return UpwardTiming(axis, counters)

    def get_attenuation(self, axis: Axis, counters: Counters, y: MakeYield):
        '''This method returns the flat atmosphere attenuation object for upward
        axes'''
        return UpwardAttenuation(axis, counters, y)

    def get_gg_file(self):
        '''This method returns the original gg array file.
        '''
        return 'gg_t_delta_theta_mc.npz'

class MakeUpwardAxisCurvedAtm(MakeUpwardAxis):
    '''This is the implementation of an upward going shower axis with a flat
    planar atmosphere'''

    def __repr__(self):
        return "UpwardAxisCurvedAtm(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
                                        self.zenith, self.azimuth, self.ground_level)

    @property
    def r(self):
        '''This is the axis distance property definition'''
        return self.h_to_axis_R_LOC(self.h, self.zenith)

    def get_timing(self, axis: Axis, counters: Counters):
        '''This method returns the instantiated upward flat atm timing object'''
        return UpwardTimingCurved(axis, counters)

    def get_attenuation(self, axis: Axis, counters: Counters, y: MakeYield):
        '''This method returns the flat atmosphere attenuation object for upward
        axes'''
        return UpwardAttenuationCurved(axis, counters, y)

    def get_gg_file(self):
        '''This method returns the original gg array file.
        '''
        return 'gg_t_delta_theta_mc.npz'

class MakeDownwardAxis(Axis):
    '''This is the implementation of an axis for a downward going shower'''

    @property
    def X(self):
        '''This method sets the depth attribute, depths are added along the axis
        in the downward direction'''
        rho = self.atm.density(self.altitude)
        axis_deltaX = np.sqrt(rho[1:] * rho[:-1]) * self.dr[1:] / 10# converting to g/cm^2
        return np.concatenate((np.cumsum(axis_deltaX[::-1])[::-1],
                    np.array([0])))

    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X[::-1], self.r[::-1])

    def theta(self, axis_vectors, counters: Counters):
        '''This method returns the angle between the axis and the vector going
        to the counter, in this case it's the internal angle'''
        return counters.calculate_theta(axis_vectors)

class MakeDownwardAxisFlatPlanarAtm(MakeDownwardAxis):
    '''This is the implementation of a downward going shower axis with a flat
    planar atmosphere.'''

    def __repr__(self):
        return "DownwardAxisFlatPlanarAtm(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.zenith, self.azimuth, self.ground_level)

    @property
    def r(self):
        '''This is the axis distance property definition'''
        return self.h / np.cos(self.zenith)

    def get_timing(self, axis: Axis, counters: Counters):
        '''This method returns the instantiated flat atm downward timing object
        '''
        return DownwardTiming(axis, counters)

    def get_attenuation(self, axis: Axis, counters: Counters, y: MakeYield):
        '''This method returns the flat atmosphere attenuation object for downward
        axes'''
        return DownwardAttenuation(axis, counters, y)

    def get_gg_file(self):
        '''This method returns the original gg array file.
        '''
        return 'gg_t_delta_theta_mc.npz'
        # return 'gg_t_delta_theta_lX_-2_to_-1.npz'

class MakeDownwardAxisCurvedAtm(MakeDownwardAxis):
    '''This is the implementation of a downward going shower axis with a
    curved atmosphere.'''

    def __repr__(self):
        return "DownwardAxisCurvedAtm(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.zenith, self.azimuth, self.ground_level)

    @property
    def r(self):
        '''This is the axis distance property definition'''
        return self.h_to_axis_R_LOC(self.h, self.zenith)

    def get_timing(self, axis: Axis, counters: Counters):
        '''This method returns the instantiated flat atm downward timing object
        '''
        return DownwardTimingCurved(axis, counters)

    def get_attenuation(self, axis: Axis, counters: Counters, y: MakeYield):
        '''This method returns the flat atmosphere attenuation object for downward
        axes'''
        return DownwardAttenuationCurved(axis, counters, y)

    def get_gg_file(self):
        '''This method returns the original gg array file.
        '''
        return 'gg_t_delta_theta_mc.npz'

def downward_curved_correction(axis: MakeDownwardAxisCurvedAtm, counters: Counters, vert: np.ndarray) -> np.ndarray:
    '''This function divides some quantity specified at each atmospheric height
    by the approriate cosine (of the local angle between vertical in the
    atmosphere and the counters), then sums those steps to the detector
    elevation. This accounts for the curvature of the earth for shower axes
    with high zenith angles. The calculation is sped up by using the cosine
    sum identity as well as interpolation between the min and max angles.

    Parameters:
    axis: instantiated MakeUpwardAxis() object
    counters: instantiated Counters() object
    vert: numpy array (same size as axis quantities) of some quantity related to
    vertically travelling photons at that stage.

    returns:
    numpy array of the corrected and integrated vertical quantity at each axis
    point going to each counter.
    The shape is:
    (# of counters, # of axis points)
    '''
    cQ = counters.cos_Q(axis.vectors)
    integrals = np.empty_like(cQ)
    Q = np.arccos(cQ)
    sQ = np.sin(Q)
    cQd = np.cos(axis.theta_difference)
    sQd = np.sin(axis.theta_difference)
    for i in range(integrals.shape[1]):
        test_Q = np.linspace(Q[:,i].min(), Q[:,i].max(), 5)
        test_cQ = np.cos(test_Q)
        test_sQ = np.sin(test_Q)
        t1 = test_cQ[:,np.newaxis] * cQd[:i] #these next three lines are what's different for up vs down
        t2 = test_sQ[:,np.newaxis] * sQd[:i]
        test_integrals = np.sum(vert[:i] / (t1 + t2), axis = 1)
        integrals[:,i] = np.interp(Q[:,i], test_Q, test_integrals)
    return integrals

def upward_curved_correction(axis: MakeUpwardAxisCurvedAtm, counters: Counters, vert: np.ndarray) -> np.ndarray:
    '''This function divides some quantity specified at each atmospheric height
    by the approriate cosine (of the local angle between vertical in the
    atmosphere and the counters), then sums those steps to the top of the
    atmosphere. This accounts for the curvature of the earth for shower axes
    with high zenith angles. The calculation is sped up by using the cosine
    sum identity as well as interpolation between the min and max angles.

    Parameters:
    axis: instantiated MakeUpwardAxis() object
    counters: instantiated Counters() object
    vert: numpy array (same size as axis quantities) of some quantity related to
    vertically travelling photons at that stage.

    returns:
    numpy array of the corrected and integrated vertical quantity at each axis
    point going to each counter.
    The shape is:
    (# of counters, # of axis points)
    '''
    cQ = counters.cos_Q(axis.vectors)
    integrals = np.empty_like(cQ)
    Q = np.arccos(cQ)
    sQ = np.sin(Q)
    cQd = np.cos(axis.theta_difference)
    sQd = np.sin(axis.theta_difference)
    for i in range(integrals.shape[1]):
        test_Q = np.linspace(Q[:,i].min(), Q[:,i].max(), 5)
        test_cQ = np.cos(test_Q)
        test_sQ = np.sin(test_Q)
        t1 = test_cQ[:,np.newaxis] * cQd[i:] #these two lines are what's different for up vs down
        t2 = test_sQ[:,np.newaxis] * sQd[i:]
        test_integrals = np.sum(vert[i:] / (t1 + t2), axis = 1)
        integrals[:,i] = np.interp(Q[:,i], test_Q, test_integrals)
    return integrals


class Timing(ABC):
    '''This is the abstract base class which contains the methods needed for
    timing photons from their source points to counting locations. Each timing
    bin corresponds to the spatial/depth bins along the shower axis.
    '''
    c = value('speed of light in vacuum')

    def __init__(self, axis: Axis, counters: Counters):
        self.axis = axis
        self.counters = counters
        # self.counter_time = self.counter_time()

    def counter_time(self) -> np.ndarray:
        '''This method returns the time it takes after the shower starts along
        the axis for each photon bin to hit each counter

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        # shower_time = self.axis_time[self.shower.profile(axis.X) > 100.]
        # return shower_time + self.travel_time + self.delay
        return self.axis_time + self.travel_time + self.delay()

    @property
    def travel_time(self) -> np.ndarray:
        '''This method calculates the the time it takes for something moving at
        c to go from points on the axis to the counters.

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        return self.counters.travel_length(self.axis.vectors) / self.c / nano

    @property
    @abstractmethod
    def axis_time(self) -> np.ndarray:
        '''This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of shape: (# of axis points,)
        '''

    @abstractmethod
    def delay(self) -> np.ndarray:
        '''This method calculates the delay a traveling photon would experience
        (compared to travelling at c) starting from given axis points to the
        detector locations

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''

class DownwardTiming(Timing):
    '''This is the implementation of timing for a downward going shower with no
    correction for atmospheric curveature
    '''

    def __init__(self, axis: MakeDownwardAxisFlatPlanarAtm, counters: Counters):
        self.axis = axis
        self.counters = counters
        # self.counter_time = self.counter_time()

    def __repr__(self):
        return f"DownwardTiming(axis=({self.axis.__repr__}), \
                                counters=({self.counters.__repr__}))"

    def vertical_delay(self):
        '''This is the delay a vertically travelling photon would experience
        compared to something travelling at c
        '''
        return np.cumsum((self.axis.delta*self.axis.dh))/self.c/nano

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return -self.axis.r / self.c / nano

    def delay(self) -> np.ndarray:
        '''This is the implementation of the delay property
        '''
        return self.vertical_delay() / self.counters.cos_Q(self.axis.vectors)

class DownwardTimingCurved(Timing):
    '''This is the implementation of timing for a downward shower using a curved
    athmosphere, this will be useful for showers with a relatively high zenith
    angle'''

    def __init__(self, axis: MakeDownwardAxisCurvedAtm, counters: Counters):
        self.axis = axis
        self.counters = counters
        # self.counter_time = self.counter_time()

    def __repr__(self):
        return f"DownwardTimingCurved(axis=({self.axis.__repr__}), \
                                      counters=({self.counters.__repr__}))"

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return -self.axis.r / self.c / nano

    def delay(self):
        '''This method returns the delay photons would experience when they
        travel from a point on the axis to a counter. The returned array is of
        shape: (# of counters, # of axis points)
        '''
        vsd = self.axis.delta * self.axis.dh / self.c / nano #vertical stage delay
        return downward_curved_correction(self.axis, self.counters, vsd)

class UpwardTiming(Timing):
    '''This is the implementation of timing for a upward going shower with no
    correction for atmospheric curveature
    '''

    def __init__(self, axis: MakeUpwardAxisFlatPlanarAtm, counters: Counters):
        self.axis = axis
        self.counters = counters
        # self.counter_time = self.counter_time()

    def __repr__(self):
        return f"UpwardTiming(axis=({self.axis.__repr__}), \
                             counters=({self.counters.__repr__}))"

    def vertical_delay(self):
        '''This is the delay a vertically travelling photon would experience
        compared to something travelling at c
        '''
        return np.cumsum((self.axis.delta*self.axis.dh)[::-1])[::-1]/self.c/nano

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return self.axis.r / self.c / nano

    def delay(self) -> np.ndarray:
        '''This is the implementation of the delay property
        '''
        return self.vertical_delay() / self.counters.cos_Q(self.axis.vectors)

class UpwardTimingCurved(Timing):
    '''This is the implementation of timing for a upward going shower with
    correction for atmospheric curveature.
    '''

    def __init__(self, axis: MakeUpwardAxisCurvedAtm, counters: Counters):
        self.axis = axis
        self.counters = counters
        # self.counter_time = self.counter_time()

    def __repr__(self):
        return f"UpwardTimingCurved(axis=({self.axis.__repr__}), \
                                   counters=({self.counters.__repr__}))"

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return self.axis.r / self.c / nano

    def delay(self):
        '''This method returns the delay photons would experience when they
        travel from a point on the axis to a counter. The returned array is of
        shape: (# of counters, # of axis points)
        '''
        vsd = self.axis.delta * self.axis.dh / self.c / nano #vertical stage delay
        return upward_curved_correction(self.axis, self.counters, vsd)

class Attenuation(ABC):
    '''This is the abstract base class whose specific implementations will
    calculate the fraction of light removed from the signal at each atmospheric
    step.
    '''
    atm = USStandardAtmosphere()

    with as_file(files('CHASM.data')/'abstable.npz') as file:
        abstable = np.load(file)

    ecoeff = abstable['ecoeff']
    l_list = abstable['wavelength']
    altitude_list = abstable['height']

    def __init__(self, axis: Axis, counters: Counters, yield_array: np.ndarray):
        self.axis = axis
        self.counters = counters
        self.yield_array = yield_array

    # def vertical_log_fraction(self) -> np.ndarray:
    #     '''This method returns the natural log of the fraction of light which
    #     survives each axis step if the light is travelling vertically.
    #
    #     The returned array is of size:
    #     # of yield bins, with each entry being on size:
    #     # of axis points
    #     '''
    #     log_fraction_array = np.empty_like(self.yield_array, dtype='O')
    #     N = self.atm.number_density(self.axis.h) / 1.e6 #convert to particles/cm^3
    #     dh = self.axis.dh * 1.e2 #convert to cm
    #     for i, y in enumerate(self.yield_array):
    #         cs = self.rayleigh_cs(self.axis.h, y.l_mid)
    #         log_fraction_array[i] = -cs * N * dh
    #     return log_fraction_array

    # def vertical_log_fraction(self) -> np.ndarray:
    #     '''This method returns the natural log of the fraction of light which
    #     survives each axis step if the light is travelling vertically.
    #
    #     The returned array is of size:
    #     # of yield bins, with each entry being of size:
    #     # of axis points
    #     '''
    #     log_fraction_array = np.empty_like(self.yield_array, dtype='O')
    #     for i, y in enumerate(self.yield_array):
    #         ecoeffs = self.ecoeff[np.abs(y.l_mid-self.l_list).argmin()]
    #         e_of_h = np.interp(self.axis.h, self.altitude_list, ecoeffs)
    #         frac_surviving = np.exp(-e_of_h)
    #         frac_step_surviving = 1. - np.diff(frac_surviving[::-1], append = 1.)[::-1]
    #         log_fraction_array[i] = np.log(frac_step_surviving)
    #     return log_fraction_array

    def vertical_log_fraction(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light which
        survives each axis step if the light is travelling vertically.

        The returned array is of size:
        # of yield bins, with each entry being of size:
        # of axis points
        '''
        return np.frompyfunc(self.calculate_vlf,1,1)(self.lambda_mids)

    def calculate_vlf(self, l):
        '''This method returns the natural log of the fraction of light which
        survives each axis step if the light is travelling vertically

        Parameters:
        y: yield object

        Returns:
        array of vertical-log-fraction values (size = # of axis points)
        '''
        ecoeffs = self.ecoeff[np.abs(l - self.l_list).argmin()]
        e_of_h = np.interp(self.axis.altitude, self.altitude_list, ecoeffs)
        frac_surviving = np.exp(-e_of_h)
        frac_step_surviving = 1. - np.diff(frac_surviving[::-1], append = 1.)[::-1]
        return np.log(frac_step_surviving)
    #

    @property
    def lambda_mids(self):
        '''This property is a numpy array of the middle of each wavelength bin'''
        l_mid_array = np.empty_like(self.yield_array)
        for i, y in enumerate(self.yield_array):
            l_mid_array[i] = y.l_mid
        return l_mid_array

    def nm_to_cm(self,l):
        return l*nano*1.e2

    def rayleigh_cs(self, h, l = 400.):
        '''This method returns the Rayleigh scattering cross section as a
        function of both the height in the atmosphere and the wavelength of
        the scattered light. This does not include the King correction factor.

        Parameters:
        h - height (m) single value or np.ndarray
        l - wavelength (nm) single value or np.ndarray

        Returns:
        sigma (cm^2 / particle)
        '''
        l_cm = self.nm_to_cm(l) #convert to cm
        N = self.atm.number_density(h) / 1.e6 #convert to particles/cm^3
        f1 = (24. * np.pi**3) / (N**2 * l_cm**4)
        n = self.atm.delta(h) + 1
        f2 = (n**2 - 1) / (n**2 + 2)
        return f1 * f2**2

    def fraction_passed(self):
        '''This method returns the fraction of light originating at each
        step on the axis which survives to reach each counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        log_fraction_passed_array = self.log_fraction_passed()
        fraction_passed_array = np.empty_like(log_fraction_passed_array, dtype= 'O')
        for i, lfp in enumerate(log_fraction_passed_array):
            fraction_passed_array[i] = np.exp(lfp)
        return fraction_passed_array

    @abstractmethod
    def log_fraction_passed(self) -> np.ndarray:
        '''This method should return the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''

class DownwardAttenuation(Attenuation):
    '''This is the implementation of signal attenuation for an downward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, axis: MakeDownwardAxisFlatPlanarAtm, counters: Counters, yield_array: np.ndarray):
        self.axis = axis
        self.counters = counters
        self.yield_array = yield_array

    def log_fraction_passed(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        vert_log_fraction_list = self.vertical_log_fraction()
        log_frac_passed_list = np.empty_like(vert_log_fraction_list, dtype='O')
        cQ = self.counters.cos_Q(self.axis.vectors)
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = np.cumsum(v_log_frac / cQ, axis=1)
        return log_frac_passed_list

class DownwardAttenuationCurved(Attenuation):
    '''This is the implementation of signal attenuation for an upward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, axis: MakeDownwardAxisCurvedAtm, counters: Counters, yield_array: np.ndarray):
        self.axis = axis
        self.counters = counters
        self.yield_array = yield_array

    def log_fraction_passed(self):
        '''This method returns the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        vert_log_fraction_list = self.vertical_log_fraction()
        log_frac_passed_list = np.empty_like(vert_log_fraction_list, dtype='O')
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = downward_curved_correction(self.axis, self.counters, v_log_frac)
        return log_frac_passed_list

class UpwardAttenuation(Attenuation):
    '''This is the implementation of signal attenuation for an upward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, axis: MakeUpwardAxisFlatPlanarAtm, counters: Counters, yield_array: np.ndarray):
        self.axis = axis
        self.counters = counters
        self.yield_array = yield_array

    def log_fraction_passed(self) -> np.ndarray:
        '''This method returns the fraction of light passed at each step from
        axis to counters.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        vert_log_fraction_list = self.vertical_log_fraction()
        log_frac_passed_list = np.empty_like(vert_log_fraction_list, dtype='O')
        cQ = self.counters.cos_Q(self.axis.vectors)
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = np.cumsum((v_log_frac / cQ)[:,::-1], axis=1)[:,::-1]
        return log_frac_passed_list

class UpwardAttenuationCurved(Attenuation):
    '''This is the implementation of signal attenuation for an upward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, axis: MakeUpwardAxisCurvedAtm, counters: Counters, yield_array: np.ndarray):
        self.axis = axis
        self.counters = counters
        self.yield_array = yield_array

    def log_fraction_passed(self):
        '''This method returns the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        vert_log_fraction_list = self.vertical_log_fraction()
        log_frac_passed_list = np.empty_like(vert_log_fraction_list, dtype='O')
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = upward_curved_correction(self.axis, self.counters, v_log_frac)
        return log_frac_passed_list
