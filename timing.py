import numpy as np
from counters import *
from axis import *
from shower import *
from scipy.constants import value,nano
from abc import ABC, abstractmethod

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
