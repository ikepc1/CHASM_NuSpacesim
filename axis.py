import numpy as np
from abc import ABC, abstractmethod
from atmosphere import Atmosphere


class Axis(ABC):
    '''This is the abstract base class which contains the methods for computing
    the cartesian vectors and corresponding slant depths of an air shower'''

    earth_radius = 6.371e6 #meters
    atm = Atmosphere()

    def __init__(self, theta: float, phi: float, ground_level: float = 0.):
        self.theta = theta
        self.phi = phi
        self.ground_level = ground_level

    @property
    def theta(self):
        '''polar angle  property getter'''
        return self._theta

    @theta.setter
    def theta(self, theta):
        '''polar angle property setter'''
        if theta > np.pi/2:
            raise ValueError('Theta cannot be greater than pi / 2')
        if theta <= 0.:
            raise ValueError('Theta cannot be less than 0')
        self._theta = theta

    @property
    def phi(self):
        '''azimuthal angle property getter'''
        return self._phi

    @phi.setter
    def phi(self, phi):
        '''azimuthal angle property setter'''
        if phi >= 2 * np.pi:
            raise ValueError('Phi must be less than 2 * pi')
        if phi < 0.:
            raise ValueError('Phi cannot be less than 0')
        self._phi = phi

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
        return np.linspace(self.ground_level+1, self.atm.maximum_height, 1000)

    @property
    def dh(self):
        '''This method sets the dr attribute'''
        dh = self.h[1:] - self.h[:-1]
        return np.concatenate((np.array([0]),dh))

    @property
    def delta(self):
        '''delta property definition'''
        return self.atm.delta(self.h)

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
        return R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)

    @property
    def r(self):
        '''r property definition'''
        return self.h_to_axis_R_LOC(self.h, self.theta)

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
        return self.theta - self.theta_normal(self.h, self.r)

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
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        cp = np.cos(self.phi)
        sp = np.sin(self.phi)
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


class MakeUpwardAxis(Axis):
    '''This is the implementation of an axis for an upward going shower, depths
    are added along the axis in the upward direction'''

    def __repr__(self):
        return "UpwardAxis(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.theta, self.phi, self.ground_level)

    @property
    def X(self):
        '''This method sets the depth attribute'''
        rho = self.atm.density(self.h)
        axis_deltaX = np.sqrt(rho[1:] * rho[:-1]) * self.dr[1:] / 10# converting to g/cm^2
        return np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))

    def distance(self, X: np.ndarray):
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X, self.r)

class MakeDownwardAxis(Axis):
    '''This is the implementation of an axis for a downward going shower'''

    def __repr__(self):
        return "DownwardAxis(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.theta, self.phi, self.ground_level)

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
