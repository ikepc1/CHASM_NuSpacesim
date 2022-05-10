import numpy as np
from abc import ABC, abstractmethod
from atmosphere import Atmosphere
from scipy.constants import value,nano
from generate_Cherenkov import MakeYield

class Axis(ABC):
    '''This is the abstract base class which contains the methods for computing
    the cartesian vectors and corresponding slant depths of an air shower'''

    earth_radius = 6.371e6 #meters
    atm = Atmosphere()

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
            raise ValueError('Azimuthal angle cannot be less than 0')
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
    def h(self):
        '''h property definition'''
        return np.linspace(self.ground_level+1., self.atm.maximum_height, 1000)

    @property
    def dh(self):
        '''This method sets the dr attribute'''
        dh = self.h[1:] - self.h[:-1]
        return np.concatenate((np.array([0]),dh))

    @property
    def delta(self):
        '''delta property definition'''
        return self.atm.delta(self.h)

    @property
    def density(self):
        '''Axis density property definition'''
        return self.atm.density(self.h)

    @classmethod
    def h_to_axis_R_LOC(cls,h,theta):
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights (m above sea level)
        theta: polar angle of shower axis (radians)

        returns: r (m) (same size as h), an array of distances along the shower
        axis_sp.
        '''
        cos_EM = np.cos(np.pi-theta)
        R = cls.earth_radius
        r_CoE= h + R # distance from the center of the earth to the specified height
        rs = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
        rs -= rs[0]
        rs[0] = 1.
        return rs # Need to find a better way to define axis zero point, currently they are all shifted by a meter to prevent divide by zero errors

    @property
    def r(self):
        '''r property definition'''
        return self.h_to_axis_R_LOC(self.h, self.zenith)

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
    def get_curved_timing(self):
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_attenuation(self):
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_curved_attenuation(self):
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''

class MakeCounters:
    '''This is the class containing the neccessary methods for finding the
    vectors from a shower axis to a user defined array of Cherenkov detectors
    with user defined size'''

    def __init__(self, input_vectors: np.ndarray, input_area: float):
        self.vectors = input_vectors
        self.area = input_area

    def __repr__(self):
        return f"Counters({self.vectors.shape[0]} counters with area = {self.area})"

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
    def area(self):
        '''Area of a counter's aperture getter'''
        return self._area

    @area.setter
    def area(self, input_area: float):
        '''Area of a counter's aperture setter'''
        if input_area <= 0.:
            raise ValueError('Counter area must be positive.')
        self._area = input_area

    @property
    def r(self):
        '''distance to each counter property definition'''
        return self.vector_magnitude(self.vectors)

    def vector_magnitude(self, vectors: np.ndarray):
        '''This method computes the length of an array of vectors'''
        return np.sqrt((vectors**2).sum(axis = -1))

    def law_of_cosines(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        '''This method returns the angle C across from side c in triangle abc'''
        cos_C = (c**2 - a**2 - b**2)/(-2*a*b)
        cos_C[cos_C > 1.] = 1.
        cos_C[cos_C < -1.] = -1.
        return np.arccos(cos_C)

    def travel_vectors(self, axis: Axis):
        '''This method returns the vectors from each entry in vectors to
        the user defined array of counters'''
        return self.vectors.reshape(-1,1,3) - axis.vectors

    def travel_length(self, axis: Axis):
        '''This method computes the distance from each point on the axis to
        each counter'''
        return self.vector_magnitude(self.travel_vectors(axis))

    def omega(self, axis: Axis):
        '''This method computes the solid angle of each counter as seen by
        each point on the axis'''
        return self.area / self.travel_length(axis)**2

    def cos_Q(self, axis: Axis):
        '''This method returns the cosine of the angle between the z-axis and
        the vector from the axis to the counter'''
        travel_n = self.travel_vectors(axis) / self.travel_length(axis)[:,:,np.newaxis]
        return np.abs(travel_n[:,:,-1])

    def travel_n(self, axis: Axis) -> np.ndarray:
        '''This method returns the unit vectors pointing along each travel
        vector.
        '''
        return self.travel_vectors(axis) / self.travel_length(axis)[:,:,np.newaxis]

    def calculate_theta(self, axis: Axis):
        '''This method calculates the angle between the axis and counters'''
        travel_length = self.travel_length(axis)
        axis_length = np.broadcast_to(axis.r, travel_length.shape)
        counter_length = np.broadcast_to(self.r, travel_length.T.shape).T
        return self.law_of_cosines(axis_length, travel_length, counter_length)

class MakeUpwardAxis(Axis):
    '''This is the implementation of an axis for an upward going shower, depths
    are added along the axis in the upward direction'''

    def __repr__(self):
        return "UpwardAxis(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.zenith, self.azimuth, self.ground_level)

    @property
    def X(self):
        '''This method sets the depth attribute'''
        rho = self.atm.density(self.h)
        axis_deltaX = np.sqrt(rho[1:] * rho[:-1]) * self.dr[1:] / 10# converting to g/cm^2
        return np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))

    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X, self.r)

    def theta(self, counters: MakeCounters):
        '''In this case we need pi minus the interal angle across from the
        distance to the counter'''
        return np.pi - counters.calculate_theta(self)

    def get_timing(self, counters: MakeCounters):
        '''This method returns the instantiated upward flat atm timing object'''
        return UpwardTiming(self, counters)

    def get_curved_timing(self, counters: MakeCounters):
        '''This method returns the instantiated upward curved atm timing object'''
        return UpwardTimingCurved(self, counters)

    def get_attenuation(self, counters: MakeCounters, y: MakeYield):
        '''This method returns the flat atmosphere attenuation object for upward
        axes'''
        return UpwardAttenuation(self, counters, y)

    def get_curved_attenuation(self, counters: MakeCounters, y: MakeYield):
        '''This method returns the curved atmosphere attenuation object for upward
        axes'''
        return UpwardAttenuationCurved(self, counters, y)

class MakeDownwardAxis(Axis):
    '''This is the implementation of an axis for a downward going shower'''

    def __repr__(self):
        return "DownwardAxis(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.zenith, self.azimuth, self.ground_level)

    @property
    def X(self):
        '''This method sets the depth attribute, depths are added along the axis
        in the downward direction'''
        rho = self.atm.density(self.h)
        axis_deltaX = np.sqrt(rho[1:] * rho[:-1]) * self.dr[1:] / 10# converting to g/cm^2
        return np.concatenate((np.cumsum(axis_deltaX[::-1])[::-1],
                    np.array([0])))

    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X[::-1], self.r[::-1])

    def theta(self, counters: MakeCounters):
        '''This method returns the angle between the axis and the vector going
        to the counter, in this case it's the internal angle'''
        return counters.calculate_theta(self)

    def get_timing(self, counters: MakeCounters):
        '''This method returns the instantiated flat atm downward timing object
        '''
        return DownwardTiming(self, counters)

    def get_curved_timing(self, counters: MakeCounters):
        '''This method returns the instantiated curved atm downward timing
        object
        '''
        return DownwardTimingCurved(self, counters)

    def get_attenuation(self, counters: MakeCounters, y: MakeYield):
        '''This method returns the flat atmosphere attenuation object for downward
        axes'''
        return DownwardAttenuation(self, counters, y)

    def get_curved_attenuation(self, counters: MakeCounters, y: MakeYield):
        '''This method returns the curved atmosphere attenuation object for downward
        axes'''
        return DownwardAttenuationCurved(self, counters, y)

def downward_curved_correction(axis: MakeDownwardAxis, counters: MakeCounters, vert: np.ndarray):
    '''This function divides some quantity specified at each atmospheric height
    by the approriate cosine (of the local angle between vertical in the
    atmosphere and the counters), then sums those steps to the detector
    elevation. This accounts for the curvature of the earth for shower axes
    with high zenith angles. The calculation is sped up by using the cosine
    sum identity as well as interpolation between the min and max angles.

    Parameters:
    axis: instantiated MakeUpwardAxis() object
    counters: instantiated MakeCounters() object
    vert: numpy array (same size as axis quantities) of some quantity related to
    vertically travelling photons at that stage.

    returns:
    numpy array of the corrected and integrated vertical quantity at each axis
    point going to each counter.
    The shape is:
    (# of counters, # of axis points)
    '''
    cQ = counters.cos_Q(axis)
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

def upward_curved_correction(axis: MakeUpwardAxis, counters: MakeCounters, vert: np.ndarray):
    '''This function divides some quantity specified at each atmospheric height
    by the approriate cosine (of the local angle between vertical in the
    atmosphere and the counters), then sums those steps to the top of the
    atmosphere. This accounts for the curvature of the earth for shower axes
    with high zenith angles. The calculation is sped up by using the cosine
    sum identity as well as interpolation between the min and max angles.

    Parameters:
    axis: instantiated MakeUpwardAxis() object
    counters: instantiated MakeCounters() object
    vert: numpy array (same size as axis quantities) of some quantity related to
    vertically travelling photons at that stage.

    returns:
    numpy array of the corrected and integrated vertical quantity at each axis
    point going to each counter.
    The shape is:
    (# of counters, # of axis points)
    '''
    cQ = counters.cos_Q(axis)
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

    def __init__(self, axis: Axis, counters: MakeCounters):
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
        return self.counters.travel_length(self.axis) / self.c / nano

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

    def __init__(self, axis: MakeDownwardAxis, counters: MakeCounters):
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
        return self.axis.r[::-1] / self.c / nano

    def delay(self) -> np.ndarray:
        '''This is the implementation of the delay property
        '''
        return self.vertical_delay() / self.counters.cos_Q(self.axis)

class DownwardTimingCurved(Timing):
    '''This is the implementation of timing for a downward shower using a curved
    athmosphere, this will be useful for showers with a relatively high zenith
    angle'''

    def __init__(self, axis: MakeDownwardAxis, counters: MakeCounters):
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
        return self.axis.r[::-1] / self.c / nano

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

    def __init__(self, axis: MakeUpwardAxis, counters: MakeCounters):
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
        return self.vertical_delay() / self.counters.cos_Q(self.axis)

class UpwardTimingCurved(Timing):
    '''This is the implementation of timing for a upward going shower with
    correction for atmospheric curveature.
    '''

    def __init__(self, axis: MakeUpwardAxis, counters: MakeCounters):
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
    atm = Atmosphere()

    def __init__(self, axis: Axis, counters: MakeCounters, yield_array: np.ndarray):
        self.axis = axis
        self.counters = counters
        self.yield_array = yield_array

    def vertical_log_fraction(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light which
        survives each axis step if the light is travelling vertically.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        # of axis points
        '''
        log_fraction_array = np.empty_like(self.yield_array, dtype='O')
        N = self.atm.number_density(self.axis.h) / 1.e6 #convert to particles/cm^3
        dh = self.axis.dh * 1.e2 #convert to cm
        for i, y in enumerate(self.yield_array):
            cs = self.rayleigh_cs(self.axis.h, y.l_mid)
            log_fraction_array[i] = -cs * N * dh
        return log_fraction_array

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

    def __init__(self, axis: MakeUpwardAxis, counters: MakeCounters, yield_array: np.ndarray):
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
        cQ = self.counters.cos_Q(self.axis)
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = np.cumsum(v_log_frac / cQ, axis=1)
        return log_frac_passed_list

class DownwardAttenuationCurved(Attenuation):
    '''This is the implementation of signal attenuation for an upward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, axis: MakeUpwardAxis, counters: MakeCounters, yield_array: np.ndarray):
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

    def __init__(self, axis: MakeUpwardAxis, counters: MakeCounters, yield_array: np.ndarray):
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
        cQ = self.counters.cos_Q(self.axis)
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = np.cumsum((v_log_frac / cQ)[:,::-1], axis=1)[:,::-1]
        return log_frac_passed_list

class UpwardAttenuationCurved(Attenuation):
    '''This is the implementation of signal attenuation for an upward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, axis: MakeUpwardAxis, counters: MakeCounters, yield_array: np.ndarray):
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
