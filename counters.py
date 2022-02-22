import numpy as np
from axis import *
from scipy.constants import value,nano
from abc import ABC, abstractmethod

class Counters(ABC):
    '''This is the abstract base class containing the neccessary methods for
    finding the vectors from a shower axis to a user defined array of Cherenkov
    detectors with user defined size'''

    def __init__(self, input_vectors: np.ndarray, input_area: float):
        self.vectors = input_vectors
        self.area = input_area

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

    @abstractmethod
    def theta(self, axis: Axis):
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''

class MakeOrbitalArray(Counters):
    '''This is the implementation of orbital Cherenkov counters'''

    def theta(self, axis: Axis):
        '''In this case we need pi minus the interal angle across from the
        distance to the counter'''
        return np.pi - self.calculate_theta(axis)

    def __repr__(self):
        return f"OrbitalArray({self.vectors.shape[0]} counters with area {self.area})"

class MakeGroundArray(Counters):
    '''This is the implementation of a ground array of Cherenkov counters'''

    def __repr__(self):
        return f"GroundArray({self.vectors.shape[0]} counters with area {self.area})"

    def theta(self, axis: Axis):
        return self.calculate_theta(axis)

class Timing(ABC):
    '''This is the abstract base class which contains the methods needed for
    timing photons from their source points to counting locations. Each timing
    bin corresponds to the spatial/depth bins along the shower axis.
    '''
    c = value('speed of light in vacuum')

    def __init__(self, axis: Axis, counters: Counters):
        self.axis = axis
        self.counters = counters
        self.counter_time = self.counter_time()

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

    def __init__(self, axis: MakeDownwardAxis, counters: Counters):
        self.axis = axis
        self.counters = counters
        self.counter_time = self.counter_time()

    def __repr__(self):
        return f"DownwardTiming(shower=({self.shower.__repr__}), \
                                axis=({self.axis.__repr__}), \
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

    def __init__(self, axis: MakeDownwardAxis, counters: Counters):
        self.axis = axis
        self.counters = counters
        self.counter_time = self.counter_time()

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return self.axis.r[::-1] / self.c / nano

    def delay(self) -> np.ndarray:
        '''This is the implementation of the delay property for a downward
        shower with a curved atmosphere

        This function calculates the delay photons experience while propogating
        through Earth's atmosphere. The calculation is sped up by using the cosine
        sum identity as well as interpolation.

        Parameters:
        axis: instantiated Axis object
        counters: instantiated Counters object

        Returns:
        The array of delays photon bunches would experience in the atmosphere. The
        array is of shape:
        (# of counters, # of axis points)
        '''
        cQ = self.counters.cos_Q(self.axis)
        delay = np.empty_like(cQ)
        Q = np.arccos(cQ)
        sQ = np.sin(Q)
        cQd = np.cos(self.axis.theta_difference)
        sQd = np.sin(self.axis.theta_difference)
        vsd = self.axis.delta * self.axis.dh /self.c/nano #vertical stage delay
        for i in range(delay.shape[1]):
            test_Q = np.linspace(Q[:,i].min(), Q[:,i].max(), 30)
            test_cQ = np.cos(test_Q)
            test_sQ = np.sin(test_Q)
            t1 = test_cQ[:,np.newaxis] * cQd[:i] #these lines are what's different between downward and upward
            t2 = test_sQ[:,np.newaxis] * sQd[:i]
            test_delay = np.sum(vsd[:i] / (t1 + t2), axis = 1)
            delay[:,i] = np.interp(Q[:,i], test_Q, test_delay)
        return delay

class UpwardTiming(Timing):
    '''This is the implementation of timing for a upward going shower with no
    correction for atmospheric curveature
    '''

    def __init__(self, axis: MakeUpwardAxis, counters: Counters):
        self.axis = axis
        self.counters = counters
        self.counter_time = self.counter_time()

    def __repr__(self):
        return f"DownwardTiming(shower=({self.shower.__repr__}), \
                                axis=({self.axis.__repr__}), \
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

    def __init__(self, axis: MakeUpwardAxis, counters: Counters):
        self.axis = axis
        self.counters = counters
        self.counter_time = self.counter_time()

    def __repr__(self):
        return f"DownwardTiming(shower=({self.shower.__repr__}), \
                                axis=({self.axis.__repr__}), \
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
        '''
        This function calculates the delay photons experience while propogating
        through Earth's atmosphere. The calculation is sped up by using the cosine
        sum identity as well as interpolation.

        Parameters:
        axis: instantiated Axis object
        counters: instantiated Counters object

        Returns:
        The array of delays photon bunches would experience in the atmosphere. The
        array is of shape:
        (# of counters, # of axis points)
        '''
        cQ = self.counters.cos_Q(self.axis)
        delay = np.empty_like(cQ)
        Q = np.arccos(cQ)
        sQ = np.sin(Q)
        cQd = np.cos(self.axis.theta_difference)
        sQd = np.sin(self.axis.theta_difference)
        vsd = self.axis.delta * self.axis.dh / self.c / nano #vertical stage delay
        for i in range(delay.shape[1]):
            test_Q = np.linspace(Q[:,i].min(), Q[:,i].max(), 5)
            test_cQ = np.cos(test_Q)
            test_sQ = np.sin(test_Q)
            t1 = test_cQ[:,np.newaxis] * cQd[i:]
            t2 = test_sQ[:,np.newaxis] * sQd[i:]
            test_delay = np.sum(vsd[i:] / (t1 + t2), axis = 1)
            delay[:,i] = np.interp(Q[:,i], test_Q, test_delay)
        return delay

class TimingFactory(ABC):
    '''This is the abstract base class for creating the timing factory'''

    @abstractmethod
    def get_timing(self):
        '''This method gets the class which does flat atmosphere timing'''

    @abstractmethod
    def get_curved_timing(self):
        '''This method gets the class which does curved atmosphere timing'''

class DownwardTimingFactory(TimingFactory):
    '''This is the implementation of the timing factory for downward-going
    showers
    '''

    def get_timing(self):
        return DownwardTiming()

    def get_curved_timing(self):
        return DownwardTimingCurved()

class UpwardTimingFactory(TimingFactory):
    '''This is the implementation of the timing factory for upward-going
    showers
    '''

    def get_timing(self):
        return UpwardTiming()

    def get_curved_timing(self):
        return UpwardTimingCurved()
