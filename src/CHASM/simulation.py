from typing import Protocol
import eventio
import numpy as np

from .shower import Shower
from .axis import Axis, Counters, MeshAxis, MeshShower
from .generate_Cherenkov import MakeYield
from .cherenkov_photon_array import CherenkovPhotonArray

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

class Signal:
    '''This class calculates the Cherenkov signal from a given shower axis at
    given counters
    '''

    def __init__(self, shower: Shower, axis: Axis, counters: Counters, yield_array: np.ndarray):
        self.shower = shower
        self.axis = axis
        self.table_file = axis.get_gg_file()
        self.gga = CherenkovPhotonArray(self.table_file)
        self.counters = counters
        self.yield_array = yield_array
        self.t = self.shower.stage(self.axis.X)
        self.t[self.t>14.] = 14.
        self.Nch = self.shower.profile(self.axis.X)
        self.theta = self.axis.theta(axis.vectors, counters)
        self.omega = self.counters.omega(self.axis.vectors)
        # self.ng = self.calculate_ng()
        # self.ng_sum = self.ng.sum(axis = 1)

    def __repr__(self):
        return f"Signal({self.shower.__repr__()}, {self.axis.__repr__()}, {self.counters.__repr__()})"

    # def calculate_gg(self):
    #     '''This funtion returns the interpolated values of gg at a given deltas
    #     and thetas

    #     returns:
    #     the angular distribution values at the desired thetas
    #     The returned array is of size:
    #     (# of counters, # of axis points)
    #     '''
    #     gg = np.empty_like(self.theta)
    #     for i in range(gg.shape[1]):
    #         gg_td = self.gga.angular_distribution(self.t[i], self.axis.delta[i])
    #         gg[:,i] = np.interp(self.theta[:,i], self.gga.theta, gg_td)
    #     return gg
    
    def calculate_gg(self):
        '''This funtion returns the interpolated values of gg at a given deltas
        and thetas

        returns:
        the angular distribution values at the desired thetas
        The returned array is of size:
        (# of counters, # of axis points)
        '''
        gg = np.empty_like(self.theta)
        for i in range(gg.shape[0]):
            gg[i] = self.gga.gg_of_t_delta_theta(self.t,self.axis.delta,self.theta[i])
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
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        gg = self.calculate_gg()
        ng_array = np.empty_like(self.yield_array, dtype='O')
        for i, y in enumerate(self.yield_array):
            y.set_yield_at_lX(self.axis.lX)
            ng_array[i] = gg * self.calculate_yield(y) * self.omega
        return ng_array

class Element(Protocol):
    '''This is the protocol for a simulation element. It needs a type, either
    axis, shower, counters, or yield'''
    @property
    def element_type(self):
        ...

    def create(self) -> object:
        ...

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
        self._has_run = False

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

    @property
    def shower(self) -> Shower:
        '''Simulation shower property'''
        return self._shower
    
    @shower.setter
    def shower(self, shower: Shower) -> None:
        self._shower = shower

    @property
    def axis(self) -> Axis:
        '''Simulation axis property'''
        return self._axis
    
    @axis.setter
    def axis(self, axis: Axis) -> None:
        self._axis = axis
    
    @property
    def y(self) -> np.ndarray[MakeYield]:
        '''Simulation yield property'''
        return self._y
    
    @y.setter
    def y(self, y: np.ndarray[MakeYield]) -> None:
        self._y = y

    @property
    def counters(self) -> Counters:
        '''Simulation counters property'''
        return self._counters
    
    @counters.setter
    def counters(self, counters: Counters) -> None:
        self._counters = counters

    def run(self, mesh: bool = False, att: bool = False):
        '''This is the proprietary run method which creates the arrays of
        Signal, Timing, and Attenuation objects
        '''
        if not self.check_ingredients():
            return None
        self.shower = self.ingredients['shower']
        self.counters = self.ingredients['counters']
        self.y = self.ingredients['yield']
        self.axis = self.ingredients['axis']
        if mesh:
            self.N_lX = len(self.lX_intervals)
            self.signals = np.empty(self.N_lX, dtype = 'O')
            self.times = np.empty_like(self.signals)
            self.attenuations = np.empty_like(self.signals)
            for i, interval in enumerate(self.lX_intervals):
                meshaxis = MeshAxis(interval,self.axis,self.shower)
                meshshower = MeshShower(meshaxis)
                self.signals[i] = Signal(meshshower,meshaxis,self.counters,self.y)
                self.times[i] = meshaxis.get_timing(self.counters)
                self.attenuations[i] = meshaxis.get_attenuation(self.counters,self.y)
                self.N_axis_points = meshaxis.r.size
                self.N_points_at_X = meshaxis.config.N_IN_RING * self.N_lX
        else:
            self.N_lX = 1
            self.signals = np.empty(1, dtype = 'O')
            self.times = np.empty(1, dtype = 'O')
            self.attenuations = np.empty(1, dtype = 'O')
            self.signals[0] = Signal(self.shower,self.axis,self.counters,self.y)
            self.times[0] = self.axis.get_timing(self.counters)
            self.attenuations[0] = self.axis.get_attenuation(self.counters,self.y)
            self.N_axis_points = self.axis.r.size
            self.N_points_at_X = 1
        self.N_c = self.counters.N_counters
        self.N_bunches = self.N_lX * self.N_axis_points
        self.signal_photons = self.get_signal_photons(att)
        self.signal_times = self.get_signal_times()
        self._has_run = True

    def get_photons_array(self, i=0):
        '''This method returns the array of photons going from each step to
        each counter for each wavelength bin.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        return self.signals[i].calculate_ng()

    def get_photons(self, i=0):
        '''This method returns the un-attenuated number of photons going from
        each step to each counter.

        The returned array is of size:
        (# of counters, # of axis points)
        '''
        photons_array = self.get_photons_array(i)
        total_photons = np.zeros_like(photons_array[0])
        for photons in photons_array:
            total_photons += photons
        return total_photons

    def get_photon_sum(self, i=0):
        '''This method returns the un-attenuated total number of photons going
        to each counter.

        The returned array is of size:
        (# of counters)
        '''
        return self.get_photons(i).sum(axis=1)

    def get_signal_sum(self):
        sum = np.zeros(self.N_c)
        for i_s in range(self.signals.size):
            sum += self.get_photon_sum(i=i_s)
        return sum

    def get_attenuated_signal_sum(self):
        sum = np.zeros(self.N_c)
        for i_s in range(self.signals.size):
            sum += self.get_attenuated_photon_sum(i=i_s)
        return sum

    def get_times(self, i=0):
        '''This method returns the time it takes after the shower starts along
        the axis for each photon bin to hit each counter. It is simply calling
        the get_times() method from a specific Timing object.

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        return self.times[i].counter_time()

    def get_photon_timebins(self, i_counter, t_min, t_max, N_bins):
        '''This method takes the arrival times of photons from each lX axis and
        puths them in the same bins whose edges are defined from t_min to t_max
        in N_bins intervals.
        '''
        hist = np.empty(N_bins)
        for i_s in range(self.signals[:,0].size):
            hist += np.histogram(self.get_times(i=i_s)[i_counter],N_bins,(t_min,t_max),weights=self.get_photons(i=i_s)[i_counter])[0]
        return hist

    # def get_signal_times(self):
    #     '''This method takes the times at which each photon bunch arrives and
    #     combines them into one array.
    #     '''
    #     N_lX = self.signals[:,0].size
    #     N_c = self.ingredients['counters'][0].N_counters
    #     times_array = np.zeros((N_c, 1))
    #     photons_array = np.zeros_like(times_array)
    #     for i_s in range(N_lX):
    #         times_array = np.append(times_array, self.get_times(i=i_s), axis = 1)
    #         photons_array = np.append(photons_array, self.get_photons(i=i_s), axis = 1)
    #     return times_array, photons_array

    def get_signal_photons(self, att = False) -> np.ndarray:
        '''This method takes the photon counts from each step from each shower
        axis object and combines them into a single array.
        '''
        photons_array = np.empty((self.N_c,self.N_bunches))
        i_s = 0
        for i_a in range(self.N_lX):
            if att:
                photons_array[:,i_s:i_s+self.N_axis_points] = self.get_attenuated_photons(i=i_a)
            else:
                photons_array[:,i_s:i_s+self.N_axis_points] = self.get_photons(i=i_a)
            i_s += self.N_axis_points
        return photons_array

    def total_ng_at_X(self) -> np.ndarray:
        '''This method returns the total number of photons going to each detector
        from each step in grammage.
        '''
        ng = np.empty_like(self.axis.X)
        iX = 0
        for i in range(ng.size):
            ng[i] = self.signal_photons[:,iX:iX+self.N_points_at_X].sum()
            iX += self.N_points_at_X
        return ng

    def get_signal_times(self) -> np.ndarray:
        '''This method takes the arrival times of photon bunches from each step
        from each shower axis object and combines them into a single array.
        '''
        times_array = np.empty((self.N_c,self.N_bunches))
        i_s = 0
        for i_a in range(self.N_lX):
            times_array[:,i_s:i_s+self.N_axis_points] = self.get_times(i=i_a)
            i_s += self.N_axis_points
        return times_array

    # def get_attenuated_signal_times(self):
    #     '''This method takes the times at which each photon bunch arrives and
    #     combines them into one array.
    #     '''
    #     times_array = np.zeros(self.N_c)
    #     photons_array = np.zeros_like(times_array)
    #     for i_s in range(self.N_lX):
    #         times_array = np.append(times_array, self.get_times(i=i_s), axis = 1)
    #         photons_array = np.append(photons_array, self.get_attenuated_photons(i=i_s), axis = 1)
    #     return times_array, photons_array

    def get_attenuated_photons_array(self, i=0):
        '''This method returns the attenuated number of photons going from each
        step to each counter.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        fraction_array = self.attenuations[i].fraction_passed()
        photons_array = self.get_photons_array(i)
        attenuated_photons = np.zeros_like(photons_array)
        for i_a, (photons, fractions) in enumerate(zip(photons_array, fraction_array)):
            attenuated_photons[i_a] = photons * fractions
        return attenuated_photons

    def get_attenuated_photons(self, i=0):
        '''This method returns the attenuated number of photons going from each
        step to each counter.

        The returned array is of size:
        (# of counters, # of axis points)
        '''
        fraction_array = self.attenuations[i].fraction_passed()
        photons_array = self.get_photons_array(i)
        attenuated_photons = np.zeros_like(photons_array[0])
        for photons, fractions in zip(photons_array, fraction_array):
            attenuated_photons += photons * fractions
        return attenuated_photons

    def get_attenuated_photon_sum(self, i=0):
        '''This method returns the attenuated total number of photons going to
        each counter.

        The returned array is of size:
        (# of counters)
        '''
        return self.get_attenuated_photons(i).sum(axis=1)
