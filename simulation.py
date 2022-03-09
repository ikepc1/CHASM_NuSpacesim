from shower import *
from axis import *
from counters import *
from generate_Cherenkov import *
# from timing import *
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Element(ABC):
    '''This is an abstract base class containing the methods needed for
    implementing a specific class in the broader simulation'''

    def convert_to_iterable(self, input_value):
        '''If one of the element arguments is a single value, it needs to be '''
        if np.size(input_value) == 1:
            return [input_value]
        else:
            return np.array(input_value)

    @abstractmethod
    def __init__(self):
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

class AxisElement(Element):
    '''This middleman class contains the setters for the theta and phi
    properties, which are the same for both axis types'''
    @property
    def theta(self):
        '''This is the phi property'''
        return self._theta

    @theta.setter
    def theta(self, input_value):
        '''If the value is a single value, convert it into a list with just
        that value, otherwise just pass a numpy array
        '''
        self._theta = self.convert_to_iterable(input_value)

    @property
    def phi(self):
        '''This is the phi property'''
        return self._phi

    @phi.setter
    def phi(self, input_value):
        '''If the value is a single value, convert it into a list with just
        that value, otherwise just pass a numpy array
        '''
        self._phi = self.convert_to_iterable(input_value)

class DownwardAxis(AxisElement):
    '''This is the implementation of the downward axis element'''
    element_type = 'axis'

    def __init__(self, theta: np.ndarray, phi: np.ndarray, ground_level: np.ndarray = 0.):
        self.theta = theta
        self.phi = phi
        self.ground_level = ground_level

    def create(self) -> MakeDownwardAxis:
        '''this method returns a dictionary element instantiated DownwardAxis
        class'''
        object_list = np.empty((np.size(self.theta), np.size(self.phi)), dtype = 'O')
        for i, t in enumerate(self.theta):
            for j, p in enumerate(self.phi):
                object_list[i, j] = MakeDownwardAxis(t, p, self.ground_level)
        return object_list

class UpwardAxis(AxisElement):
    '''This is the implementation of the downward axis element'''
    element_type = 'axis'

    def __init__(self, theta: float, phi: float, ground_level: float = 0):
        self.theta = theta
        self.phi = phi
        self.ground_level = ground_level

    def create(self) -> MakeDownwardAxis:
        '''this method returns a dictionary element instantiated DownwardAxis
        class'''
        object_list = np.empty((np.size(self.theta), np.size(self.phi)), dtype = 'O')
        for i, t in enumerate(self.theta):
            for j, p in enumerate(self.phi):
                object_list[i, j] = MakeUpwardAxis(t, p, self.ground_level)
        return object_list

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
        return self.convert_to_iterable(MakeGHShower(self.X_max, self.N_max, self.X0, self.Lambda))

class UserShower(Element):
    '''This is the implementation of the GH shower element'''
    element_type = 'shower'

    def __init__(self,X: np.ndarray, Nch: np.ndarray):
        self.X = X
        self.Nch = Nch

    def create(self):
        '''This method returns an instantiated user shower '''
        return self.convert_to_iterable(MakeUserShower(self.X, self.Nch))

class OrbitalArray(Element):
    '''This is the implementation of the orbital array element'''
    element_type = 'counters'

    def __init__(self, input_vectors: np.ndarray, input_area: float):
        self.vectors = input_vectors
        self.area = input_area

    def create(self):
        '''This method returns an instantiated orbital array'''
        return self.convert_to_iterable(MakeOrbitalArray(self.vectors, self.area))

class GroundArray(Element):
    '''This is the implementation of the ground array element'''
    element_type = 'counters'

    def __init__(self, input_vectors: np.ndarray, input_area: float):
        self.vectors = input_vectors
        self.area = input_area

    def create(self):
        '''This method returns an instantiated orbital array'''
        return self.convert_to_iterable(MakeGroundArray(self.vectors, self.area))

class Yield(Element):
    '''This is the implementation of the yield element'''
    element_type = 'yield'

    def __init__(self, l_min: float, l_max: float):
        self.l_min = l_min
        self.l_max = l_max

    def create(self):
        '''This method returns an instantiated yield object'''
        return self.convert_to_iterable(MakeYield(self.l_min, self.l_max))

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
        # self.ng = self.calculate_ng()
        # self.ng_sum = self.ng.sum(axis = 1)

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
        'axis': None,
        'shower': None,
        'counters': None,
        'yield': None
        }

    def add(self, element: Element):
        '''Add a element to the list of elements for the sim to perform'''
        self.ingredients[element.element_type] = element.create()

    def remove(self, type):
        '''Remove all ingredients of a certain type from simulation'''
        self.ingredients[type] = None

    def check_ingredients(self) -> bool:
        '''This method checks to see if the simulation has the neccesary
        elements to generate a Cherenkov signal.
        '''
        for element_type in self.ingredients:
            if type(self.ingredients[element_type]) == None:
                print(f"Simulation needs {element_type}")
                return False
        return True

    def run(self, curved = False):
        shower = self.ingredients['shower'][0]
        counters = self.ingredients['counters'][0]
        y = self.ingredients['yield'][0]
        axis = self.ingredients['axis']
        self.signals = np.empty_like(self.ingredients['axis'])
        self.times = np.empty_like(self.ingredients['axis'])
        if self.check_ingredients():
            for i in range(axis.shape[0]):
                for j in range(axis.shape[1]):
                    self.signals[i,j] = Signal(shower, axis[i,j], counters, y)
                    if curved:
                        self.times[i,j] = counters.get_timing_factory().get_curved_timing()(axis[i,j],counters)
                    else:
                        self.times[i,j] = counters.get_timing_factory().get_timing()(axis[i,j],counters)

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

    # theta = np.linspace(.01, np.radians(80),100)
    # phi = np.linspace(0, 1.999*np.pi, 10)

    # x = np.linspace(0,10000,11)
    # xx, yy = np.meshgrid(x,x)
    # counters = np.empty([xx.size,3])
    # counters[:,0] = xx.flatten()
    # counters[:,1] = yy.flatten()
    # counters[:,2] = np.zeros(xx.size)

    # counters = np.zeros([30,3])
    # counters[:,0] = np.logspace(-1,3,30)

    counters = np.empty([100,3])

    theta = np.radians(85)
    phi = 0.
    r = 2141673.2772862054

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    counters[:,0] = np.full(100,x)
    # counters[:,0] = np.linspace(-1000,1000,100)
    counters[:,1] = np.linspace(y-100.e3,y+100.e3,100)
    counters[:,2] = np.full(100,z)

    area = 0.03141593

    sim = ShowerSimulation()
    sim.add(UpwardAxis(theta,phi))
    sim.add(GHShower(666.,6e7,0.,70.))
    sim.add(OrbitalArray(counters, area))
    sim.add(Yield(300,450))
    # sim.plot_profile()
    sim.run(curved = True)

    # plt.figure()
    # plt.plot(sim.ingredients['axis'][0].r,sim.timing.delay()[0], label = 'flat atm')
    # plt.plot(sim.ingredients['axis'][0].r,sim.timing_curved.delay()[0], label = 'curved atm')
    # plt.legend()
    #
    # plt.figure()
    # plt.scatter(counters[:,0],sim.signal.ng_sum)
    # plt.loglog()
    # plt.grid()
