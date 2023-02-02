import numpy as np
from scipy.constants import value,nano
from cherenkov_photon import CherenkovPhoton
from cherenkov_photon_array import CherenkovPhotonArray
from counter_array import CounterArray as gc
from functools import lru_cache

class CherenkovYield(gc):
    '''A class for calculating the Cherenkov yield of an upward going air shower
    '''
    c = value('speed of light in vacuum')
    hc = value('Planck constant in eV s') * c
    gga = CherenkovPhotonArray('gg_t_delta_theta_2020_normalized.npz')
    # gga = CherenkovPhotonArray('one_direction_table.npz')

    def __init__(self,X_max,N_max,Lambda,X0,theta,direction,tel_vectors,min_l,max_l,split):
        super().__init__(X_max,N_max,Lambda,X0,theta,direction,tel_vectors,split)
        self.tel_dE = self.hc/(min_l*nano) - self.hc/(max_l*nano)

        if self.direction == 'up':
            self.axis_time = self.axis_r/self.c/nano
            self.delay_prime = self.calculate_delay(self.cQ, self.i_ch, self.theta_difference, self.axis_delta, self.axis_dh)
            self.counter_time_prime = self.axis_time[self.i_ch] + self.travel_length/self.c/nano + self.delay_prime
            if split:
                self.split_axis_time = self.split_axis_r/self.c/nano
                self.split_delay_prime = self.calculate_delay(self.split_cQ, self.split_i_ch, self.split_theta_difference, self.split_axis_delta, self.split_axis_dh)
                self.split_counter_time_prime = self.split_axis_time[self.split_i_ch] + self.split_travel_length/self.c/nano + self.split_delay_prime
        else:
            self.axis_time = self.axis_r[::-1]/self.c/nano
            self.delay_prime = self.calculate_delay(self.cQ, self.i_ch, self.theta_difference, self.axis_delta, self.axis_dh)
            self.counter_time_prime = self.axis_time[self.i_ch] + self.travel_length/self.c/nano + self.delay_prime

        self.delay = self.calculate_vertical_delay(self.axis_delta,self.axis_dh)[self.i_ch]/self.cQ
        self.counter_time = self.axis_time[self.i_ch] + self.travel_length/self.c/nano + self.delay
        self.ng, self.ng_sum, self.gg = self.calculate_ng(self.shower_t,
                self.shower_delta,self.tel_q,self.shower_nch,self.shower_dr,
                self.tel_omega,self.tel_dE)
        if split:
            self.split_delay = self.calculate_vertical_delay(self.split_axis_delta,self.split_axis_dh)[self.split_i_ch]/self.split_cQ
            self.split_counter_time = self.split_axis_time[self.split_i_ch] + self.split_travel_length/self.c/nano + self.split_delay
            self.split_ng, self.split_ng_sum, self.split_gg = self.calculate_ng(self.split_shower_t,
                    self.split_shower_delta,self.split_tel_q,self.split_shower_nch,self.split_shower_dr,
                    self.split_tel_omega,self.tel_dE)



    @lru_cache
    def interpolate_gg(self,t,delta,theta):
        '''This funtion returns the interpolated values of gg at a given delta
        and theta
        parameters:
        t: single value of the stage
        delta: single value of the delta
        theta: array of theta values at which we want to return the angular
        distribution

        returns:
        the angular distribution values at the desired thetas
        '''

        gg_td = self.gga.angular_distribution(t,delta)
        return np.interp(theta,self.gga.theta,gg_td)

    def calculate_gg(self, t, delta, theta):
        '''This funtion returns the interpolated values of gg at a given deltas
        and thetas
        parameters:
        t: array of stages
        delta: array of corresponding deltas
        theta: array of theta values at each stage

        returns:
        the angular distribution values at the desired thetas
        '''
        gg = np.empty_like(theta)
        for i in range(theta.shape[1]):
            gg_td = self.gga.angular_distribution(t[i],delta[i])
            gg[:,i] = np.interp(theta[:,i],self.gga.theta,gg_td)
        return gg

    def calculate_yield(self,delta,shower_nch,shower_dr,tel_dE):
        ''' This function returns the total number of Cherenkov photons emitted
        at a given stage of a shower per all solid angle.

        Parameters:
        delta: single value or array of the differences from unity of the index of
        refraction (n-1)
        shower_nch: single value or array of the number of charged particles.
        shower_dr: single value or array, the spatial distance represented by
        the given stage
        lw: the wavelength of the lower bound of the Cherenkov energy interval
        hw: the wavelength of the higher bound of the Cherenkov energy interval

        returns: the total number of photons per all solid angle
        '''

        alpha_over_hbarc = 370.e2 # per eV per m, from PDG
        c = value('speed of light in vacuum')
        hc = value('Planck constant in eV s') * c
        chq = CherenkovPhoton.cherenkov_angle(1.e12,delta)
        cy = alpha_over_hbarc*np.sin(chq)**2*tel_dE
        return shower_nch * shower_dr * cy

    def calculate_ng(self,t,delta,theta,shower_nch,shower_dr,omega,tel_dE):
        gg = self.calculate_gg(t,delta,theta)
        ng = gg * omega * self.calculate_yield(delta,shower_nch,shower_dr,tel_dE)
        return ng, ng.sum(axis = 1), gg

    def calculate_vertical_delay(self,axis_delta,axis_dh):
        if self.direction == 'up':
            delay = np.cumsum((axis_delta*axis_dh)[::-1])[::-1]/self.c/nano
        else:
            delay = np.cumsum((axis_delta*axis_dh))/self.c/nano
        return delay

    # @lru_cache
    # def calculate_delay(self):
    #     delay = np.empty_like(self.cQ)
    #     i_min = np.min(self.i_ch)
    #     sQ = np.sin(np.arccos(self.cQ))
    #     cQd = np.cos(self.theta_difference)
    #     sQd = np.sin(self.theta_difference)
    #     vsd = self.axis_delta*self.axis_dh
    #     for i in range(delay.shape[0]):
    #         for j in range(delay.shape[1]):
    #             if self.direction == 'up':
    #                 cos = self.cQ[i,j]*cQd[i_min + j:] + sQ[i,j]*sQd[i_min + j:] # cosine difference identity
    #                 delay[i,j] = np.sum(vsd[i_min + j:]/cos)
    #             else:
    #                 cos = self.cQ[i,j]*cQd[0:i_min + j] + sQ[i,j]*sQd[0:i_min + j]
    #                 delay[i,j] = np.sum(vsd[0:i_min + j]/cos)
    #     return delay/self.c/nano

    # @jit(nopython=True)
    def calculate_delay(self, cQ, i_ch, theta_difference, axis_delta, axis_dh):
        '''
        This function calculates the delay photons experience while propogating
        through Earth's atmosphere.

        Parameters:
        cQ = array of cosines of the angles each photon bunch's travel path makes with the z axis
        i_ch = indices along the full axis where there are significant charged particles
        theta_difference = array of corrections to theta along the axis to make theta at that
        point be with respect to vertical in the atmosphere, not the z axis
        axis_delta = array of atmospheric deltas along the whole axis
        axis_dh = array of height differences along the whole axis
        Returns:
        The array of delays (same shape as cQ) photon bunches would experience in the atmosphere.
        '''
        delay = np.empty_like(cQ)
        i_min = np.min(i_ch)
        Q = np.arccos(cQ)
        sQ = np.sin(Q)
        cQd = np.cos(theta_difference)
        sQd = np.sin(theta_difference)
        vsd = axis_delta*axis_dh
        for i in range(delay.shape[1]):
            test_Q = np.linspace(Q[:,i].min(),Q[:,i].max(),5)
            test_cQ = np.cos(test_Q)
            test_sQ = np.sin(test_Q)
            t1 = test_cQ[:,np.newaxis]*cQd[i_min + i:]
            t2 = test_sQ[:,np.newaxis]*sQd[i_min + i:]
            test_delay = np.sum(vsd[i_min + i:]/(t1 + t2), axis = 1)
            delay[:,i] = np.interp(Q[:,i],test_Q,test_delay)
        return delay/self.c/nano
