from shower import *
from axis import *
from counters import *
from generate_Cherenkov import *
from timing import *
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Element(ABC):
    '''This is an abstract base class containing the methods needed for
    implementing a specific class in the broader simulation'''

    @abstractmethod
    def __init__(self, *args, **kwargs):
        '''The element implementation should recieve the user arguments that the
        element itself needs and its __init__ should save them as attributes'''

    @property
    @abstractmethod
    def element_type(self):
        '''This is the element type property. It needs to be a string, either
        axis, shower, or counters.
        '''

    @abstractmethod
    def create(self):
        '''This method instantiates the specific element in question'''

class DownwardAxis(Element):
    '''This is the implementation of the downward axis element'''
    element_type = 'axis'

    def __init__(self, theta: float, phi: float, ground_level: float = 0):
        self.theta = theta
        self.phi = phi
        self.ground_level = ground_level

    def create(self) -> MakeDownwardAxis:
        '''this method returns a dictionary element instantiated DownwardAxis
        class'''
        return MakeDownwardAxis(self.theta, self.phi, self.ground_level)

class UpwardAxis(Element):
    '''This is the implementation of the downward axis element'''
    element_type = 'axis'

    def __init__(self, theta: float, phi: float, ground_level: float = 0):
        self.theta = theta
        self.phi = phi
        self.ground_level = ground_level

    def create(self) -> MakeDownwardAxis:
        '''this method returns a dictionary element instantiated DownwardAxis
        class'''
        return MakeUpwardAxis(self.theta, self.phi, self.ground_level)

class GHShower(Element):
    '''This is the implementation of the GH shower element'''
    element_type = 'shower'

    def __init__(self, X_max: float, N_max: float, X0: float, Lambda: float):
        self.X_max = X_max
        self.N_max = N_max
        self.X0 = X0
        self.Lambda = Lambda

    def create(self):
        '''This method returns an instantiated Gaisser Hillas Shower '''
        return MakeGHShower(self.X_max, self.N_max, self.X0, self.Lambda)

class UserShower(Element):
    '''This is the implementation of the GH shower element'''
    element_type = 'shower'

    def __init__(self,X: np.ndarray, Nch: np.ndarray):
        self.X = X
        self.Nch = Nch

    def create(self):
        '''This method returns an instantiated user shower '''
        return MakeUserShower(self.X, self.Nch)

class OrbitalArray(Element):
    '''This is the implementation of the orbital array element'''
    element_type = 'counters'

    def __init__(self, input_vectors: np.ndarray, input_area: float):
        self.vectors = input_vectors
        self.area = input_area

    def create(self):
        '''This method returns an instantiated orbital array'''
        return MakeOrbitalArray(self.vectors, self.area)

class GroundArray(Element):
    '''This is the implementation of the ground array element'''
    element_type = 'counters'

    def __init__(self, input_vectors: np.ndarray, input_area: float):
        self.vectors = input_vectors
        self.area = input_area

    def create(self):
        '''This method returns an instantiated orbital array'''
        return MakeGroundArray(self.vectors, self.area)

class Yield(Element):
    '''This is the implementation of the yield element'''
    element_type = 'yield'

    def __init__(self, l_min: float, l_max: float):
        self.l_min = l_min
        self.l_max = l_max

    def create(self):
        '''This method returns an instantiated yield object'''
        return MakeYield(self.l_min, self.l_max)

class Signal:
    '''This class calculates the Cherenkov signal from a given shower axis at
    given counters
    '''
    table_file = 'gg_t_delta_theta_doubled.npz'
    gga = CherenkovPhotonArray(table_file)

    def __init__(self, shower: Shower, axis: Axis, counters: Counters, y: MakeYield):
        self.shower = shower
        self.axis = axis
        self.counters = counters
        self.y = y #because "yield" is python boiler-plate
        self.t = self.shower.stage(axis.X)
        self.Nch = self.shower.profile(axis.X)
        self.theta = self.counters.theta(axis)
        self.omega = self.counters.omega(axis)
        self.ng = self.calculate_ng()
        self.ng_sum = self.ng.sum(axis = 1)

    def calculate_gg(self):
        '''This funtion returns the interpolated values of gg at a given deltas
        and thetas

        returns:
        the angular distribution values at the desired thetas
        The returned array is of size:
        (# of counters, # of axis points)
        '''
        gg = np.empty_like(self.theta)
        for i in range(gg.shape[1]):
            gg_td = self.gga.angular_distribution(self.t[i], self.axis.delta[i])
            gg[:,i] = np.interp(self.theta[:,i], self.gga.theta, gg_td)
        return gg

    def calculate_yield(self):
        ''' This function returns the total number of Cherenkov photons emitted
        at a given stage of a shower per all solid angle.

        returns: the total number of photons per all solid angle
        size: (# of axis points)
        '''
        Y = self.y.y_list(self.t, self.axis.delta)
        return self.Nch * self.axis.dr * Y

    def calculate_ng(self):
        '''This method returns the number of Cherenkov photons going toward
        each counter from every axis bin

        The returned array is of size:
        (# of counters, # of axis points)
        '''
        return self.calculate_gg() * self.calculate_yield() * self.omega


class ShowerSimulation:
    '''This class is the framework for creating a simulation'''

    def __init__(self):
        self.ingredients = {
        'axis': [],
        'shower': [],
        'counters': [],
        'yield': []
        }

    def add(self, element: Element):
        '''Add a element to the list of elements for the sim to perform'''
        self.ingredients[element.element_type].append(element.create())

    def remove(self, type):
        '''Remove all ingredients of a certain type from simulation'''
        self.ingredients[type] = []

    def check_ingredients(self) -> bool:
        '''This method checks to see if the simulation has the neccesary
        elements to generate a Cherenkov signal.
        '''
        for element_type in self.ingredients:
            if self.ingredients[element_type] == []:
                print(f"Simulation needs {element_type}")
                return False
        return True

    def run(self):
        if self.check_ingredients():
            shower = self.ingredients['shower'][0]
            axis = self.ingredients['axis'][0]
            counters = self.ingredients['counters'][0]
            y = self.ingredients['yield'][0]
            self.timing_curved = DownwardTimingCurved(axis, counters)
            self.timing = DownwardTiming(axis, counters)
            self.signal = Signal(shower, axis, counters, y)

    def plot_profile(self):
        a = self.ingredients['axis'][0]
        s = self.ingredients['shower'][0]
        plt.ion()
        plt.figure()
        plt.plot(a.X, s.profile(a.X))


if __name__ == '__main__':
    import numpy as np
    from shower import *
    plt.ion()

    theta = np.radians(80)
    phi = np.pi

    # x = np.linspace(0,10000,11)
    # xx, yy = np.meshgrid(x,x)
    # counters = np.empty([xx.size,3])
    # counters[:,0] = xx.flatten()
    # counters[:,1] = yy.flatten()
    # counters[:,2] = np.zeros(xx.size)

    counters = np.zeros([30,3])
    counters[:,0] = np.logspace(-1,3,30)

    # counters = np.empty([100,3])
    #
    # r = 2141673.2772862054
    #
    # x = r * np.sin(theta) * np.cos(phi)
    # y = r * np.sin(theta) * np.sin(phi)
    # z = r * np.cos(theta)
    #
    # counters[:,0] = np.full(100,x)
    # # counters[:,0] = np.linspace(-1000,1000,100)
    # counters[:,1] = np.linspace(y-100.e3,y+100.e3,100)
    # counters[:,2] = np.full(100,z)

    area = 0.03141593

    sim = ShowerSimulation()
    sim.add(DownwardAxis(theta,phi))
    sim.add(GHShower(666.,6e7,0.,70.))
    sim.add(GroundArray(counters, area))
    sim.add(Yield(300,450))
    # sim.plot_profile()
    sim.run()

    plt.figure()
    plt.plot(sim.ingredients['axis'][0].r,sim.timing.delay()[0], label = 'flat atm')
    plt.plot(sim.ingredients['axis'][0].r,sim.timing_curved.delay()[0], label = 'curved atm')
    plt.legend()

    plt.figure()
    plt.scatter(counters[:,0],sim.signal.ng_sum)
    plt.loglog()
    plt.grid()
