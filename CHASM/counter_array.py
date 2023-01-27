import numpy as np
from scipy.constants import value,nano
from scipy.stats import norm
from shower_axis import Shower as sh
from functools import lru_cache

class CounterArray(sh):
    '''Class for calculating Cherenkov yield of upward going showers at a
    hypothetical orbital telescope array_z

    Parameters:
    shower_r: array of distances along shower axis (m)
    n_tel: number of telescopes
    tel_distance: how far they are along the axis from first interaction point (m)
    tel_area: surface area of telescopes (m^2)

    '''

    def __init__(self,X_max,N_max,Lambda,X0,theta,direction,tel_vectors,split):
        super().__init__(X_max,N_max,Lambda,X0,theta,direction)
        if tel_vectors.shape[1] != 3 or len(tel_vectors.shape) != 2:
            raise Exception("tel_vectors is not an array of vectors.")
        self.reset_array(tel_vectors,split)

    def reset_array(self,tel_vectors,split):
        self.axis_vectors = self.set_axis_vectors(self.shower_r, self.theta, self.phi)
        self.tel_vectors = tel_vectors
        self.tel_area = 1
        self.tel_q, self.tel_omega, self.travel_length, self.cQ = self.set_travel_params(self.axis_vectors,self.tel_vectors,self.tel_area)
        if split:
            self.spread_axis(10)
            self.split_tel_q, self.split_tel_omega, self.split_travel_length, self.split_cQ = self.set_travel_params(self.split_axis,self.tel_vectors,self.tel_area)
            self.split_stages()
            self.distribute_nch(.25)


    def set_axis_vectors(self,shower_r,theta, phi):
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        axis_vectors = np.empty([np.shape(shower_r)[0],3])
        axis_vectors[:,0] = shower_r * st * cp
        axis_vectors[:,1] = shower_r * st * sp
        axis_vectors[:,2] = shower_r * ct
        return axis_vectors

    def set_travel_params(self,axis_vectors,tel_vectors,tel_area):
        travel_vectors = tel_vectors.reshape(-1,1,3) - axis_vectors
        travel_length =  np.sqrt( (travel_vectors**2).sum(axis=2) )
        axis_r = np.sqrt( (axis_vectors**2).sum(axis=1) )
        tel_r = np.sqrt( (tel_vectors**2).sum(axis=1) )

        travel_n = travel_vectors/travel_length[:,:,np.newaxis]
        travel_cQ = np.abs(travel_n[:,:,-1])
        axis_length = np.broadcast_to(axis_r,travel_length.shape)
        tel_length = np.broadcast_to(tel_r,travel_length.T.shape).T

        cq = (tel_length**2-axis_length**2-travel_length**2)/(-2*axis_length*travel_length) #cosine of angle between axis and vector
        cq[cq>1.] = 1.
        cq[cq<-1.] = -1.
        tel_q = np.arccos(cq)
        if self.direction == 'up':
            tel_q = np.pi - tel_q
        tel_omega = tel_area / travel_length **2
        return tel_q, tel_omega, travel_length, travel_cQ

    # @jit(nopython=True)
    def spread_axis(self, n):
        '''
        n = the number of lateral points in which to spread the axis
        '''
        self.n = n
        self.repeat = 2*n
        self.split_axis = np.empty((self.axis_vectors.shape[0] * self.repeat, self.axis_vectors.shape[1]))
        rm_cq_cp = self.shower_rms_w * self.axis_cq * self.axis_cp
        rm_cq_sp = self.shower_rms_w * self.axis_cq * self.axis_sp
        rm_sq = self.shower_rms_w * self.axis_sq
        rm_sp = self.shower_rms_w * self.axis_sp
        rm_cp = self.shower_rms_w * self.axis_cp
        zeros = np.zeros_like(rm_cp)
        scaler = np.linspace(-1,1,n)
        self.shift_vector_1 = np.array([-rm_cq_cp, -rm_cq_sp, rm_sq]).T
        self.shift_vector_2 = np.array([rm_sp, -rm_cp, zeros]).T
        for i in range(n):
            self.split_axis[i::self.repeat] = self.axis_vectors + scaler[i]*self.shift_vector_1
            self.split_axis[i+n::self.repeat] = self.axis_vectors + scaler[i]*self.shift_vector_2

    def split_stages(self):
        self.split_axis_r = np.repeat(self.axis_r,self.repeat)
        split_total_axis_nch = np.repeat(self.axis_nch,self.repeat)
        self.split_i_ch = np.nonzero(split_total_axis_nch)[0]
        self.split_theta_difference = np.repeat(self.theta_difference,self.repeat)
        self.split_axis_delta = np.repeat(self.axis_delta,self.repeat)
        self.split_axis_dh = np.repeat(self.axis_dh,self.repeat)
        self.split_shower_t = np.repeat(self.shower_t,self.repeat)
        self.split_shower_delta = np.repeat(self.shower_delta,self.repeat)
        self.split_shower_dr = np.repeat(self.shower_dr,self.repeat)

    def distribute_nch(self, sig):
        scaler = np.linspace(-1,1,self.n)
        nch_scaler = norm.pdf(scaler, scale = sig)
        nch_scaler /= nch_scaler.sum()
        split_nch = np.repeat(self.shower_nch/2,2)[:,np.newaxis] * nch_scaler
        self.split_shower_nch = split_nch.flatten()
