import numpy as np
from axis import Axis
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
