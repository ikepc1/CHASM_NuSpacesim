import eventio
import numpy as np
from atmosphere import *
from scipy.constants import value,nano
from scipy.stats import norm
import sys
from CHASM.simulation import *

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
        self.set_longitudinal_info()

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

    def set_longitudinal_info(self):
        iact_l = self.event.longitudinal
        nl = iact_l['nthick']
        self.iact_X = np.arange(nl,dtype=float)*iact_l['thickstep']
        self.iact_nch = np.array(iact_l['data'][6])
        self.iact_Xmax = self.iact_X[self.iact_nch.argmax()]
        self.iact_t = (self.iact_X-self.iact_Xmax)/36.62

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

class IACTShower(eventio.IACTFile):
    """This class generates a shower profile from an event in a CORSIKA iact file """
    c = value('speed of light in vacuum')
    hc = value('Planck constant in eV s') * c
    X0 = 36.62 # Radiation length in air
    atm = USStandardAtmosphere()

    def __init__(self, infile):
        super().__init__(infile)
        self.event_list()
        # self.atm = at.Atmosphere()
        self.atm_h, self.atm_X = self.set_atm_depths()
        self.set_CORSIKA_params()
        self.percs_and_bins()
        self.set_longitudinal_info()
        self.interpolate_height()
        self.set_axis_vectors()
        self.iact_nch[self.iact_axis_pos[:,2] < 0.] = 0

    def event_list(self):
        self.event_list = []
        for event in self:
            self.event_list.append(event)
        if len(self.event_list) == 1:
            self.ev = self.event_list[0]

    def set_atm_depths(self):
        '''
        Get atmosphere and make an X table for interpolating X values to height
        '''
        atm_h = (np.arange(np.ceil(self.atm.maximum_height/100))*100)[::-1] # Reverse ordered heights in 100m steps
        atm_h2 = np.roll(atm_h,1)
        atm_h2[0] = atm_h[0]+100
        atm_deltaX = self.atm.depth(atm_h,atm_h2)
        atm_X = np.cumsum(atm_deltaX)
        return atm_h, atm_X

    def set_CORSIKA_params(self):
        iact_counter_x = self.telescope_positions['x']/100. # cm -> m
        iact_counter_y = self.telescope_positions['y']/100. # cm -> m
        iact_counter_z = self.telescope_positions['z']/100. # cm -> m
        self.iact_counter_r = self.telescope_positions['r']/100. # cm -> m
        self.iact_counter_A = np.pi*self.iact_counter_r**2
        self.iact_nc = len(iact_counter_x)
        self.iact_obs = self.header[5][4]/100. # The 6th entry in the header are the observation levels; there's only one.
        self.iact_counter_pos = np.empty((iact_counter_x.shape[0],3),dtype=float)
        self.iact_counter_pos[:,0] = iact_counter_x
        self.iact_counter_pos[:,1] = iact_counter_y
        self.iact_counter_pos[:,2] = iact_counter_z
        self.min_l = self.ev.header[57]
        self.max_l = self.ev.header[58]
        self.iact_dE = self.hc/(self.min_l*nano) - self.hc/(self.max_l*nano)
        self.iact_theta = self.ev.header[10]
        self.iact_phi = self.ev.header[11] + np.pi
        self.iact_cq = np.cos(self.iact_theta)
        self.iact_sq = np.sin(self.iact_theta)
        self.iact_cp = np.cos(self.iact_phi)
        self.iact_sp = np.sin(self.iact_phi)
        self.iact_axis_n = np.array((-self.iact_sq*self.iact_cp,-self.iact_sq*self.iact_sp,-self.iact_cq))
        self.iact_ng_sum = np.array([self.ev.n_photons[i] for i in range(self.iact_nc)])

    def percs_and_bins(self):
        iact_gmnt = np.array([self.ev.photon_bunches[i]['time'].min() for i in range(self.iact_nc)])
        iact_gmxt = np.array([self.ev.photon_bunches[i]['time'].max() for i in range(self.iact_nc)])
        iact_g05t = np.array([np.percentile(self.ev.photon_bunches[i]['time'], 5.) for i in range(self.iact_nc)])
        iact_g95t = np.array([np.percentile(self.ev.photon_bunches[i]['time'],95.) for i in range(self.iact_nc)])
        iact_gd90 = iact_g95t-iact_g05t
        iact_g01t = np.array([np.percentile(self.ev.photon_bunches[i]['time'], 1.) for i in range(self.iact_nc)])
        iact_g99t = np.array([np.percentile(self.ev.photon_bunches[i]['time'],99.) for i in range(self.iact_nc)])
        self.wl = np.abs(np.array([self.ev.photon_bunches[i]['wavelength'] for i in range(self.iact_nc)]))
        iact_ghdt = np.ones_like(iact_gmnt)
        iact_ghdt[iact_gd90<30.]   = 0.2
        iact_ghdt[iact_gd90>100.] = 5.
        self.iact_ghmn = np.floor(iact_gmnt)
        self.iact_ghmn[iact_ghdt==5.] = 5*np.floor(self.iact_ghmn[iact_ghdt==5.]/5.)
        self.iact_ghmx = np.ceil(iact_gmxt)
        self.iact_ghmx[iact_ghdt==5.] = 5*np.ceil(self.iact_ghmx[iact_ghdt==5.]/5.)
        self.iact_ghnb = ((self.iact_ghmx-self.iact_ghmn)/iact_ghdt).astype(int)

    def set_longitudinal_info(self):
        iact_l = self.ev.longitudinal
        nl = iact_l['nthick']
        self.iact_X = np.arange(nl,dtype=float)*iact_l['thickstep']
        self.iact_nch = np.array(iact_l['data'][6])
        self.iact_Xmax = self.iact_X[self.iact_nch.argmax()]
        self.iact_t = (self.iact_X-self.iact_Xmax)/self.X0

    def interpolate_height(self):
        vert_X = self.iact_X * self.iact_cq
        self.iact_h = np.interp(vert_X,self.atm_X,self.atm_h)
        self.iact_delta = self.atm.delta(self.iact_h)
        iact_midh = np.sqrt(self.iact_h[1:]*self.iact_h[:-1])
        self.iact_dh = np.empty_like(self.iact_h)
        self.iact_dh[1:-1] = iact_midh[:-1]-iact_midh[1:]
        self.iact_dh[-1] = self.iact_dh[-2]
        self.iact_dh[0] = self.iact_dh[1]
        self.iact_dl = self.iact_dh/self.iact_cq

    def set_axis_vectors(self):
        self.iact_r = (self.iact_h-self.iact_obs)/self.iact_cq
        self.iact_axis_pos = np.empty((self.iact_h.shape[0],3),dtype=float)
        self.iact_axis_pos[:,0] = self.iact_r*self.iact_sq*self.iact_cp
        self.iact_axis_pos[:,1] = self.iact_r*self.iact_sq*self.iact_sp
        self.iact_axis_pos[:,2] = self.iact_h-self.iact_obs

    def calculate_delay(self):
        self.iact_vertical_delay = np.cumsum((self.iact_delta*self.iact_dh)[::-1])[::-1]/self.c/nano
        self.iact_axis_time = -self.iact_r/self.c/nano

def sim_from_cors(sh: IACTShower):
    sim = ShowerSimulation()
    sim.add(DownwardAxis(sh.iact_theta,sh.iact_phi,sh.iact_obs))
    sim.add(UserShower(sh.iact_X,sh.iact_nch))
    sim.add(SphericalCounters(sh.iact_counter_pos, sh.iact_counter_r))
    sim.add(Yield(sh.min_l,sh.max_l,10))
    sim.run(mesh=True)
    return sim

if __name__ == '__main__':
    import numpy as np
    from CHASM.simulation import *
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    plt.ion()
    import eventio
    
    corsika_file = '/home/isaac/corsika_data/unattenuated_30degree_sealevel/iact_s_000011.dat'
    att_corsika_file = '/home/isaac/corsika_data/attenuated_30degree_sealevel/iact_s_000011.dat'
    # corsika_file = 'IACT_shower_noatt.dat'
    # att_corsika_file = 'IACT_shower_att.dat'
    corsika_file = '/home/isaac/corsika_data/correct_thinning/iact_DAT000001.dat'
    # # corsika_file = '/home/isaac/corsika_data/oldbuild_thinned/iact_DAT000001.dat'
    # corsika_file = '/home/isaac/corsika_data/old_build_diff_seed/iact_DAT000003.dat'

    cors_no_att = EventioWrapper(corsika_file) #CORSIKA shower with no atmospheric absorbtion
    cors_att = EventioWrapper(att_corsika_file) #CORSIKA shower with atmospheric absorbtion

    sim = ShowerSimulation()

    #Add shower axis
    sim.add(DownwardAxis(cors_no_att.theta, cors_no_att.phi, cors_no_att.obs))

    #Add shower profile
    sim.add(UserShower(cors_no_att.X, cors_no_att.nch)) #profiles could also be created using Gaisser-Hillas parameters

    #Add CORSIKA IACT style spherical telescopes
    sim.add(SphericalCounters(cors_no_att.counter_vectors, cors_no_att.counter_radius))

    #Add wavelength interval for Cherenkov yield calculation
    sim.add(Yield(cors_no_att.min_l, cors_no_att.max_l))

    #Create objects which calculate Cherekov signal
    sim.run(mesh = True)

    #Get signal photons
    ng_sum = sim.get_signal_sum()

    #Get attenuated signal photons
    ng_sum_att = sim.get_attenuated_signal_sum()

    fig = plt.figure()
    plt.scatter(cors_no_att.counter_r, cors_no_att.ng_sum, c = 'k', label = 'CORSIKA IACT');
    plt.scatter(cors_no_att.counter_r, cors_att.ng_sum, c = 'g', label = 'CORSIKA IACT (Attenuated)');
    plt.scatter(cors_no_att.counter_r, ng_sum, c = 'r', label = 'CHASM');
    plt.scatter(cors_no_att.counter_r, ng_sum_att, c = 'b', label = 'CHASM (Attenuated)');
    plt.semilogy();
    plt.legend();
    plt.grid()
    plt.title('Cherenkov Lateral Distribution');

    sim.run(mesh = False)
    times = sim.get_signal_times()
    photons = sim.get_signal_photons()
    sim.run(mesh = True)
    # mesh_times, mesh_photons = sim.get_attenuated_signal_times()
    mesh_times = sim.get_signal_times()
    mesh_photons = sim.get_signal_photons()

    # counters_2_plot = np.array([1,5,10,20,30,40,45]) #indices of counters for which to plot arrival time distributions
    counters_2_plot = np.array([0,1,2,5,8,10,12,15,20,25,30,31,32,35,40,45])
    # counters_2_plot = np.arange(0,20)
    for counter in counters_2_plot:
        fig = plt.figure()
        plt.hist(cors_no_att.get_photon_times(counter), cors_no_att.iact_ghnb[counter], (cors_no_att.iact_ghmn[counter],cors_no_att.iact_ghmx[counter]),
                            color='k',
                            weights=cors_no_att.get_photons(counter),
                            histtype='step',label='CORSIKA-IACT')
        plt.hist(mesh_times[counter], cors_no_att.iact_ghnb[counter], (cors_no_att.iact_ghmn[counter],cors_no_att.iact_ghmx[counter]),
                        color='b',
                        weights=mesh_photons[counter],
                        histtype='step',label='CHASM mesh')
        plt.hist(times[counter], cors_no_att.iact_ghnb[counter], (cors_no_att.iact_ghmn[counter],cors_no_att.iact_ghmx[counter]),
                        color='r',
                        weights=photons[counter],
                        histtype='step',label='CHASM no mesh')
        plt.xlabel('Time (ns)')
        plt.ylabel('Number of Cherenkov photons')
        plt.title(f'Counter {cors_no_att.counter_r[counter]:.1f} m from Core.')
        plt.semilogx()
        plt.legend()


    