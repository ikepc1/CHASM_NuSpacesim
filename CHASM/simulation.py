from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import eventio

from .shower import *
from .axis import *
from .generate_Cherenkov import *

#Wrap eventio for extraction of CORSIKA shower data.
class EventioWrapper(eventio.IACTFile):
    def __init__(self, corsika_filename):
        super().__init__(corsika_filename)
        self.event = self.get_event()
        self.theta = self.event.header[10]
        self.phi = self.event.header[11] + np.pi #CHASM coordinate system rhat points up the axis, not down like CORSIKA
        self.obs = self.event.header[5]/100. #convert obs level to meters
        nl = self.event.longitudinal['nthick'] #number of depth steps
        self.X = np.arange(nl, dtype=float) * self.event.longitudinal['thickstep'] #create depth steps
        self.nch = np.array(self.event.longitudinal['data'][6]) #corresponding number of charged particles
        counter_x = self.telescope_positions['x']/100. # cm -> m
        counter_y = self.telescope_positions['y']/100. # cm -> m
        counter_z = self.telescope_positions['z']/100. # cm -> m
        self.counter_r = np.sqrt(counter_x**2 + counter_y**2 + counter_z**2)
        self.counter_radius = self.telescope_positions['r']/100. # cm -> m
        self.counter_vectors = np.vstack((counter_x,counter_y,counter_z)).T
        self.iact_nc = len(self.counter_r)
        self.min_l = self.event.header[57] #wavelength in nm
        self.max_l = self.event.header[58] #wavelength in nm
        self.ng_sum = np.array([self.event.n_photons[i] for i in range(len(counter_x))])
        self.percs_and_bins(self.event)

    def get_event(self, event_index: int = 0):
        '''This method returns an individual shower event from the CORSIKA IACT file.'''
        event_list = []
        for event in self:
            event_list.append(event)
        if len(event_list) == 1:
            return event_list[0]
        else:
            return event_list[event_index]

    def percs_and_bins(self, event):
        pb = []
        for i in range(self.iact_nc):
            if np.size(event.photon_bunches[i]['time']) == 0:
                pb.append(np.array([0.]))
            else:
                pb.append(event.photon_bunches[i]['time'])
        iact_gmnt = np.array([pb[i].min() for i in range(self.iact_nc)])
        iact_gmxt = np.array([pb[i].max() for i in range(self.iact_nc)])
        iact_g05t = np.array([np.percentile(pb[i], 5.) for i in range(self.iact_nc)])
        iact_g95t = np.array([np.percentile(pb[i],95.) for i in range(self.iact_nc)])
        iact_gd90 = iact_g95t-iact_g05t
        iact_g01t = np.array([np.percentile(pb[i], 1.) for i in range(self.iact_nc)])
        iact_g99t = np.array([np.percentile(pb[i],99.) for i in range(self.iact_nc)])
        iact_ghdt = np.ones_like(iact_gmnt)
        iact_ghdt[iact_gd90<30.]   = 0.2
        iact_ghdt[iact_gd90>100.] = 5.
        self.iact_ghmn = np.floor(iact_gmnt)
        self.iact_ghmn[iact_ghdt==5.] = 5*np.floor(self.iact_ghmn[iact_ghdt==5.]/5.)
        self.iact_ghmx = np.ceil(iact_gmxt)
        self.iact_ghmx[iact_ghdt==5.] = 5*np.ceil(self.iact_ghmx[iact_ghdt==5.]/5.)
        self.iact_ghnb = ((self.iact_ghmx-self.iact_ghmn)/iact_ghdt).astype(int)
        self.iact_ghnb[self.iact_ghnb == 0] = 1

    def get_photon_times(self, counter_index: int, event_index: int = 0):
        '''This method returns the array of arrival times for each photon bunch for a particular
        event and counter.'''
        return self.get_event(event_index).photon_bunches[counter_index]['time']

    def get_photons(self, counter_index: int, event_index: int = 0):
        '''This method returns the array of the number of photons in each photon bunch for a particular
        event and counter.'''
        return self.get_event(event_index).photon_bunches[counter_index]['photons']

    def shower_coordinates(self):
        """
        This function returns the emmission coordinates of CORSIKA photon bunches
        and the number of photons at each emission coordinate (for each IACT)
        """
        x_e = []
        y_e = []
        z_e = []
        p_e = []
        obslevel = self.header[5][4]
        for i in range(len(self.telescope_positions)):
            pos = self.telescope_positions[i]
            x_t = np.array(self.event.photon_bunches[i]['x'])
            y_t = np.array(self.event.photon_bunches[i]['y'])
            z = np.array(self.event.photon_bunches[i]['zem'])
            cx = np.array(self.event.photon_bunches[i]['cx'])
            cy = np.array(self.event.photon_bunches[i]['cy'])
            p = np.array(self.event.photon_bunches[i]['photons'])
            z -= obslevel
            cz = np.sqrt(1 - (cx**2 + cy**2))
            x = x_t - cx * z / cz + pos[0]
            y = y_t - cy * z / cz + pos[1]
            x_e = np.append(x_e,x / 100)
            y_e = np.append(y_e,y / 100)
            z_e = np.append(z_e,z / 100)
            p_e = np.append(p_e,p)
        return np.array(x_e),np.array(y_e),np.array(z_e),np.array(p_e)

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
    def zenith(self):
        '''This is the phi property'''
        return self._zenith

    @zenith.setter
    def zenith(self, input_value):
        '''If the value is a single value, convert it into a list with just
        that value, otherwise just pass a numpy array
        '''
        self._zenith = self.convert_to_iterable(input_value)

    @property
    def azimuth(self):
        '''This is the phi property'''
        return self._azimuth

    @azimuth.setter
    def azimuth(self, input_value):
        '''If the value is a single value, convert it into a list with just
        that value, otherwise just pass a numpy array
        '''
        self._azimuth = self.convert_to_iterable(input_value)

class DownwardAxis(AxisElement):
    '''This is the implementation of the downward axis element'''
    element_type = 'axis'

    def __init__(self, zenith: float, azimuth: float, ground_level: float = 0, curved: bool = False):
        self.zenith = zenith
        self.azimuth = azimuth
        self.ground_level = ground_level
        self.curved = curved

    def create(self) -> np.ndarray:
        '''this method returns a dictionary element instantiated DownwardAxis
        class'''
        object_list = np.empty((np.size(self.zenith), np.size(self.azimuth)), dtype = 'O')
        for i, t in enumerate(self.zenith):
            for j, p in enumerate(self.azimuth):
                if self.curved:
                    object_list[i, j] = MakeDownwardAxisCurvedAtm(t, p, self.ground_level)
                else:
                    object_list[i, j] = MakeDownwardAxisFlatPlanarAtm(t, p, self.ground_level)
        return object_list

class UpwardAxis(AxisElement):
    '''This is the implementation of the downward axis element'''
    element_type = 'axis'

    def __init__(self, zenith: float, azimuth: float, ground_level: float = 0, curved: bool = False):
        self.zenith = zenith
        self.azimuth = azimuth
        self.ground_level = ground_level
        self.curved = curved

    def create(self) -> np.ndarray:
        '''this method returns a dictionary element instantiated DownwardAxis
        class'''
        object_list = np.empty((np.size(self.zenith), np.size(self.azimuth)), dtype = 'O')
        for i, t in enumerate(self.zenith):
            for j, p in enumerate(self.azimuth):
                if self.curved:
                    object_list[i, j] = MakeUpwardAxisCurvedAtm(t, p, self.ground_level)
                else:
                    object_list[i, j] = MakeUpwardAxisFlatPlanarAtm(t, p, self.ground_level)
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

class SphericalCounters(Element):
    '''This is the implementation of the ground array element'''
    element_type = 'counters'

    def __init__(self, input_vectors: np.ndarray, input_radius: float):
        self.vectors = input_vectors
        self.radius = input_radius

    def create(self):
        '''This method returns an instantiated orbital array'''
        return self.convert_to_iterable(MakeSphericalCounters(self.vectors, self.radius))

class FlatCounters(Element):
    '''This is the implementation of the ground array element'''
    element_type = 'counters'

    def __init__(self, input_vectors: np.ndarray, input_radius: float):
        self.vectors = input_vectors
        self.radius = input_radius

    def create(self):
        '''This method returns an instantiated orbital array'''
        return self.convert_to_iterable(MakeFlatCounters(self.vectors, self.radius))

class Yield(Element):
    '''This is the implementation of the yield element'''
    element_type = 'yield'

    def __init__(self, l_min: float, l_max: float, N_bins: int = 1):
        self.l_min = l_min
        self.l_max = l_max
        self.N_bins = N_bins

    def make_lambda_bins(self):
        '''This method creates a list of bin low edges and a list of bin high
        edges'''
        bin_edges = np.linspace(self.l_min, self.l_max, self.N_bins+1)
        return bin_edges[:-1], bin_edges[1:]

    def create(self):
        '''This method returns an instantiated yield object'''
        bin_minimums, bin_maximums = self.make_lambda_bins()
        yield_array = np.empty_like(bin_minimums, dtype = 'O')
        for i, (min, max) in enumerate(zip(bin_minimums, bin_maximums)):
            yield_array[i] = MakeYield(min, max)
        return yield_array

class Signal:
    '''This class calculates the Cherenkov signal from a given shower axis at
    given counters
    '''
    # table_file = 'gg_t_delta_theta_doubled.npz'
    # gga = CherenkovPhotonArray(table_file)

    def __init__(self, shower: Shower, axis: Axis, counters: Counters, yield_array: np.ndarray):
        self.shower = shower
        self.axis = axis
        self.table_file = axis.get_gg_file()
        self.gga = CherenkovPhotonArray(self.table_file)
        self.counters = counters
        self.yield_array = yield_array
        self.t = self.shower.stage(self.axis.X)
        self.Nch = self.shower.profile(self.axis.X)
        self.theta = self.axis.theta(axis.vectors, counters)
        self.omega = self.counters.omega(self.axis.vectors)
        # self.ng = self.calculate_ng()
        # self.ng_sum = self.ng.sum(axis = 1)

    def __repr__(self):
        return f"Signal({self.shower.__repr__()}, {self.axis.__repr__()}, {self.counters.__repr__()})"

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

    def calculate_yield(self, y: MakeYield):
        ''' This function returns the total number of Cherenkov photons emitted
        at a given stage of a shower per all solid angle.

        returns: the total number of photons per all solid angle
        size: (# of axis points)
        '''
        Y = y.y_list(self.t, self.axis.delta)
        return 2. * self.Nch * self.axis.dr * Y

    def calculate_ng(self):
        '''This method returns the number of Cherenkov photons going toward
        each counter from every axis bin

        The returned array is of size:
        (# of counters, # of axis points)
        '''
        gg = self.calculate_gg()
        ng_array = np.empty_like(self.yield_array, dtype='O')
        for i, y in enumerate(self.yield_array):
            y.set_yield_at_lX(self.axis.lX)
            ng_array[i] = gg * self.calculate_yield(y) * self.omega
        return ng_array

class ShowerSimulation:
    '''This class is the framework for creating a simulation'''
    # lXs = np.arange(-4,1)
    lXs = np.linspace(-6,1,15)
    lX_intervals = list(zip(lXs[:-1], lXs[1:]))
    # lXs = np.linspace(-4,1,5)

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

    # def run(self, mesh: bool = False):
    #     '''This is the proprietary run method which creates the arrays of
    #     Signal, Timing, and Attenuation objects
    #     '''
    #     shower = self.ingredients['shower'][0]
    #     counters = self.ingredients['counters'][0]
    #     y = self.ingredients['yield']
    #     axis = self.ingredients['axis']
    #     self.signals = np.empty_like(self.ingredients['axis'])
    #     self.times = np.empty_like(self.ingredients['axis'])
    #     self.attenuations = np.empty_like(self.ingredients['axis'])
    #     if self.check_ingredients():
    #         for i in range(axis.shape[0]):
    #             for j in range(axis.shape[1]):
    #                 if mesh:
    #                     a = MeshAxis(axis[i,j],shower)
    #                     s = MeshShower(a)
    #                 else:
    #                     a = axis[i,j]
    #                     s = shower
    #                 self.signals[i,j] = Signal(s, a, counters, y)
    #                 self.times[i,j] = a.get_timing(a, counters)
    #                 self.attenuations[i,j] = a.get_attenuation(a, counters, y)

    def run(self, mesh: bool = False):
        '''This is the proprietary run method which creates the arrays of
        Signal, Timing, and Attenuation objects
        '''
        if not self.check_ingredients():
            return None
        shower = self.ingredients['shower'][0]
        counters = self.ingredients['counters'][0]
        y = self.ingredients['yield']
        axis = self.ingredients['axis'][0,0]
        if mesh:
            lX_intervals = list(zip(self.lXs[:-1], self.lXs[1:]))
            self.signals = np.empty((len(lX_intervals),1), dtype = 'O')
            self.times = np.empty((len(lX_intervals),1), dtype = 'O')
            self.attenuations = np.empty((len(lX_intervals),1), dtype = 'O')
            for i, interval in enumerate(lX_intervals):
                meshaxis = MeshAxis(interval,axis,shower)
                meshshower = MeshShower(meshaxis)
                self.signals[i,0] = Signal(meshshower,meshaxis,counters,y)
                self.times[i,0] = meshaxis.get_timing(meshaxis, counters)
                self.attenuations[i,0] = meshaxis.get_attenuation(meshaxis, counters,y)
        else:
            self.signals = np.empty((1,1), dtype = 'O')
            self.times = np.empty((1,1), dtype = 'O')
            self.attenuations = np.empty((1,1), dtype = 'O')
            self.signals[0,0] = Signal(shower,axis,counters,y)
            self.times[0,0] = axis.get_timing(axis, counters)
            self.attenuations[0,0] = axis.get_attenuation(axis, counters,y)


    def plot_profile(self):
        a = self.ingredients['axis'][0]
        s = self.ingredients['shower'][0]
        plt.ion()
        plt.figure()
        plt.plot(a.X, s.profile(a.X))

    def get_photons_array(self, i=0, j=0):
        '''This method returns the array of photons going from each step to
        each counter for each wavelength bin.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        return self.signals[i,j].calculate_ng()

    def get_photons(self, i=0, j=0):
        '''This method returns the un-attenuated number of photons going from
        each step to each counter.

        The returned array is of size:
        (# of counters, # of axis points)
        '''
        photons_array = self.get_photons_array(i,j)
        total_photons = np.zeros_like(photons_array[0])
        for photons in photons_array:
            total_photons += photons
        return total_photons

    def get_photon_sum(self, i=0, j=0):
        '''This method returns the un-attenuated total number of photons going
        to each counter.

        The returned array is of size:
        (# of counters)
        '''
        return self.get_photons(i,j).sum(axis=1)

    def get_signal_sum(self):
        sum = np.zeros(self.ingredients['counters'][0].r.size)
        for i_s in range(self.signals[:,0].size):
            sum += self.get_photon_sum(i=i_s)
        return sum

    def get_attenuated_signal_sum(self):
        sum = np.zeros(self.ingredients['counters'][0].r.size)
        for i_s in range(self.signals[:,0].size):
            sum += self.get_attenuated_photon_sum(i=i_s)
        return sum

    def get_times(self, i=0, j=0):
        '''This method returns the time it takes after the shower starts along
        the axis for each photon bin to hit each counter. It is simply calling
        the get_times() method from a specific Timing object.

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        return self.times[i,j].counter_time()

    def get_photon_timebins(self, i_counter, t_min, t_max, N_bins):
        '''This method takes the arrival times of photons from each lX axis and
        puths them in the same bins whose edges are defined from t_min to t_max
        in N_bins intervals.
        '''
        hist = np.empty(N_bins)
        for i_s in range(self.signals[:,0].size):
            hist += np.histogram(self.get_times(i=i_s)[i_counter],N_bins,(t_min,t_max),weights=self.get_photons(i=i_s)[i_counter])[0]
        return hist

    def get_signal_times(self):
        '''This method takes the times at which each photon bunch arrives and
        combines them into one array.
        '''
        N_lX = self.signals[:,0].size
        N_c = self.ingredients['counters'][0].N_counters
        times_array = np.zeros((N_c, 1))
        photons_array = np.zeros_like(times_array)
        for i_s in range(N_lX):
            times_array = np.append(times_array, self.get_times(i=i_s), axis = 1)
            photons_array = np.append(photons_array, self.get_photons(i=i_s), axis = 1)
        return times_array, photons_array

    def get_attenuated_signal_times(self):
        '''This method takes the times at which each photon bunch arrives and
        combines them into one array.
        '''
        N_lX = self.signals[:,0].size
        N_c = self.ingredients['counters'][0].N_counters
        times_array = np.zeros((N_c, 1))
        photons_array = np.zeros_like(times_array)
        for i_s in range(N_lX):
            times_array = np.append(times_array, self.get_times(i=i_s), axis = 1)
            photons_array = np.append(photons_array, self.get_attenuated_photons(i=i_s), axis = 1)
        return times_array, photons_array

    def get_attenuated_photons_array(self, i=0, j=0):
        '''This method returns the attenuated number of photons going from each
        step to each counter.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        fraction_array = self.attenuations[i,j].fraction_passed()
        photons_array = self.get_photons_array(i,j)
        attenuated_photons = np.zeros_like(photons_array)
        for i, (photons, fractions) in enumerate(zip(photons_array, fraction_array)):
            attenuated_photons[i] = photons * fractions
        return attenuated_photons

    def get_attenuated_photons(self, i=0, j=0):
        '''This method returns the attenuated number of photons going from each
        step to each counter.

        The returned array is of size:
        (# of counters, # of axis points)
        '''
        fraction_array = self.attenuations[i,j].fraction_passed()
        photons_array = self.get_photons_array(i,j)
        attenuated_photons = np.zeros_like(photons_array[0])
        for photons, fractions in zip(photons_array, fraction_array):
            attenuated_photons += photons * fractions
        return attenuated_photons

    def get_attenuated_photon_sum(self, i=0, j=0):
        '''This method returns the attenuated total number of photons going to
        each counter.

        The returned array is of size:
        (# of counters)
        '''
        return self.get_attenuated_photons(i,j).sum(axis=1)

    def get_attenuated_signal_sum(self):
        sum = np.zeros(self.ingredients['counters'][0].r.size)
        for i_s in range(self.signals[:,0].size):
            sum += self.get_attenuated_photon_sum(i=i_s)
        return sum

if __name__ == '__main__':
    import numpy as np
    from shower import *
    plt.ion()

    # theta = np.linspace(.01, np.radians(80),100)
    # phi = np.linspace(0, 1.999*np.pi, 10)
    theta = np.radians(50)
    phi = np.radians(135)

    x = np.linspace(-1000,1000,100)
    xx, yy = np.meshgrid(x,x)
    counters = np.empty([xx.size,3])
    counters[:,0] = xx.flatten()
    counters[:,1] = yy.flatten()
    counters[:,2] = np.zeros(xx.size)

    sim = ShowerSimulation()
    sim.add(DownwardAxis(theta,phi))
    sim.add(GHShower(666.,6e7,0.,70.))
    sim.add(FlatCounters(counters, 1.))
    sim.add(Yield(200,205,1))
    sim.run(mesh=True)

    fig = plt.figure()
    h2d = plt.hist2d(counters[:,0],counters[:,1],weights=sim.get_signal_sum(),bins=100, cmap=plt.cm.jet)
    # plt.title('Cherenkov Upward Shower footprint at ~500km')
    plt.xlabel('Counter Plane X-axis (km)')
    plt.ylabel('Counter Plane Y-axis (km)')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(label = 'Number of Cherenkov Photons')

    # axis =  sim.ingredients['axis'][0,0]
    # shower = sim.ingredients['shower'][0]
    # counters = sim.ingredients['counters'][0]
    # y = sim.ingredients['yield']
    # ma = MeshAxis(axis, shower)
    # ms = MeshShower(ma)
    # # mesh_axis, r, t, d = axis_to_mesh(axis,shower)
    # # rotated_mesh_axis = rotate_mesh(mesh_axis, axis.zenith, axis.azimuth)
    # ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # ax.scatter(ma.rotated_mesh[:,0],ma.rotated_mesh[:,1],ma.rotated_mesh[:,2],s=.1)
    # ax.scatter(ma.vectors[:,0],ma.vectors[:,1],ma.vectors[:,2],c=ma.nch/ma.nch.max())
    # ax.scatter(axis.vectors[:,0],axis.vectors[:,1],axis.vectors[:,2],s=.5)
    # ax.set_xlim(-1000,1000)
    # ax.set_ylim(-1000,1000)
    # ax.set_zlim(0,6000)
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('z (m)')


    # counters = np.empty([100,3])
    #
    # theta = np.radians(85)
    # phi = 0.
    # r = 2141673.2772862054
    #
    # x = r * np.sin(theta) * np.cos(phi)
    # y = r * np.sin(theta) * np.sin(phi)
    # z = r * np.cos(theta)
    #
    # counters[:,0] = np.full(100,x)
    # counters[:,1] = np.linspace(y-100.e3,y+100.e3,100)
    # counters[:,2] = np.full(100,z)
    #
    # area = 1
    #
    # sim = ShowerSimulation()
    # sim.add(UpwardAxis(theta,phi))
    # sim.add(GHShower(666.,6e7,0.,70.))
    # sim.add(Counters(counters, area))
    # sim.add(Yield(300,450))
    # sim.run(curved = True)
    #
    # s = sim.signals[0,0]
    # ng = s.calculate_ng()
    # plt.figure()
    # plt.plot(s.counters.vectors[:,1],ng.sum(axis=1),label='no attenuation')
    # ng_att = ng * s.axis.get_attenuation(s.counters, s.y).fraction_passed()
    # ng_att_curved = ng * s.axis.get_curved_attenuation(s.counters, s.y).fraction_passed()
    # plt.plot(s.counters.vectors[:,1],ng_att.sum(axis=1), label='flat attenuation')
    # plt.plot(s.counters.vectors[:,1],ng_att_curved.sum(axis=1), label='curved attenuation')
    # plt.legend()
