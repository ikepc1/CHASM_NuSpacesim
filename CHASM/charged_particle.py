"""
This module contains classes for creating the energy and angular
distributions of charged particles.
These were originally written by Zane Gerber, September 2019.
Modified by Douglas Bergman, September 2019.
Modified to include Douglas Bergman's 2013 parameterization, June 2020
"""

import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import quad

from .atmosphere import *

class EnergyDistribution:
    """
    This class contains functions related to the energy distribution
    of secondary particles.  The parameterizations used are those of
    S. Lafebre et al. (2009). The normalization parameter A1 is determined
    by the normalization condition.
    """
    # pt for particle
    pt = {'Tot': 0, 'Ele': 1, 'Pos': 2}
    # pm for parameter
    pm = {'A00':0,'A01':1,'A02':2,'e11':3,'e12':4,'e21':5,'e22':6,'g11':7,'g21':8}
    # pz for parameterization
    #               A00   A01   A02     e11  e12    e21  e22  g11 g21
    pz = np.array([[1.000,0.191,6.91e-4,5.64,0.0663,123.,0.70,1.0,0.0374],  # Tot
                   [0.485,0.183,8.17e-4,3.22,0.0068,106.,1.00,1.0,0.0372],  # Ele
                   [0.516,0.201,5.42e-4,4.36,0.0663,143.,0.15,2.0,0.0374]]) # Pos

    ll = np.log(1.e-1) #lower limit
    ul = np.log(1.e11) #upper limit

    def __init__(self,part,t):
        """
        Set the parameterization constants for this type of particle. The normalization
        constant is determined for the given shower stage, (which can be changed later).
        Parameters:
            particle = The name of the distribution of particles to create
            t = The shower stage for which to do the claculation
        """
        self.p = self.pt[part]
        self.t = t
        self.normalize(t)

    # Functions for the top level parameters
    def _set_A0(self,p,t):
        self.A0 = self.A1*self.pz[p,self.pm['A00']] * np.exp( self.pz[p,self.pm['A01']]*t - self.pz[p,self.pm['A02']]*t**2)
    def _set_e1(self,p,t):
        self.e1 = self.pz[p,self.pm['e11']] - self.pz[p,self.pm['e12']]*t
    def _set_e2(self,p,t):
        self.e2 = self.pz[p,self.pm['e21']] - self.pz[p,self.pm['e22']]*t
    def _set_g1(self,p,t):
        self.g1 = self.pz[p,self.pm['g11']]
    def _set_g2(self,p,t):
        self.g2 = 1 + self.pz[p,self.pm['g21']]*t

    def normalize(self,t):
        p = self.pt['Tot']
        self.A1 = 1.
        self._set_A0(p,t)
        self._set_e1(p,t)
        self._set_e2(p,t)
        self._set_g1(p,t)
        self._set_g2(p,t)
        intgrl,eps = quad(self.spectrum,self.ll,self.ul)
        self.A1 = 1/intgrl
        p = self.p
        self._set_A0(p,t)
        self._set_e1(p,t)
        self._set_e2(p,t)
        self._set_g1(p,t)
        self._set_g2(p,t)

    def set_stage(self,t):
        self.t = t
        self.normalize(t)

    def spectrum(self,lE):
        """
        This function returns the particle distribution as a function of energy (energy spectrum)
        at a given stage
        Parameters:
            lE = energy of a given secondary particle [MeV]
        Returns:
            n_t_lE = the energy distribution of secondary particles.
        """
        E = np.exp(lE)
        return self.A0*E**self.g1 / ( (E+self.e1)**self.g1 * (E+self.e2)**self.g2 )

class AngularDistribution:
    """
    This class contains functions related to the angular distribution
    of secondary particles.  This class can produce an electron angular
    distribution based on either the parameterization of Lafebre et al. or
    Professor Bergman depending on the choice of the variable 'schema'
    """
    # Bergman constants
    pm_b = {
        'a10' : 3773.05,
        'a11' : 1.82945,
        'a12' : 0.031143,
        'a13' : 0.0129724,
        'c10' : 163.366,
        'c11' : 0.952228,
        'c20' : 182.945,
        'c21' : 0.921291,
        'a20' : 340.308,
        'a21' : 1.73569,
        'a22' : 6.03581,
        'a23' : 4.29495,
        'a24' : 2.50626,
        'p0'  : 49.0374,
        'p1'  : 0.790002,
        'p2'  : 2.20173,
        'r0'  : 3.6631,
        'r1'  : 0.131998,
        'r2'  : -0.134479,
        'r3'  : 0.537966,
        'lb'  : -1.5,
        'lc'  : -1.4,
    }
    # Lafebre constants
    pm_l = {
        'a11' : -0.399,
        'a21' : -8.36,
        'a22' : 0.440,
        'sig' : 3,
        'b11' : -3.73,
        'b12' : 0.92,
        'b13' : 0.210,
        'b21' : 32.9,
        'b22' : 4.84,
    }

    intlim = np.array([0,1.e-11,1.e-9,1.e-7,1.e-5,1.e-3,1.e-1,np.pi])
    lls = intlim[:-1]
    uls = intlim[1:]

    def __init__(self,lE,schema='b'):
        """Set the parameterization constants for this type (log)energy. The
        angular distribution only depends on the energy not the
        particle or stage. The normalization constanct is determined
        automatically. (It's normalized in degrees!)
        Parameters:
            lE = The log of the energy (in MeV) at which the angular
                 distribution is calculated
            schema = either 'b' for the Bergman parameterization or 'l' for the
                     Lafebre parameterization.
        """
        self.schema = schema
        self.set_lE(lE)

    # Set Lafebre constants
    def _set_b1l(self):
        self.b1l = self.pm_l['b11'] + self.pm_l['b12'] * self.E**self.pm_l['b13']
    def _set_b2l(self):
        self.b2l = self.pm_l['b21'] - self.pm_l['b22'] * self.lE
    def _set_a1l(self):
        self.a1l = self.pm_l['a11']
    def _set_a2l(self):
        self.a2l = self.pm_l['a21'] + self.pm_l['a22'] * self.lE
    def _set_sigl(self):
        self.sigl = self.pm_l['sig']

    # Set Bergman constants
    def _set_a1b(self):
        self.a1b = self.pm_b['a10'] * (self.EGeV)**(self.pm_b['a11'] +
        self.pm_b['a12'] * self.log10E + self.pm_b['a13'] *
        self.log10E**2)
    def _set_c1b(self):
        self.c1b = self.pm_b['c10'] * (self.EGeV)**self.pm_b['c11']
    def _set_c2b(self):
        self.c2b = self.pm_b['c20'] * (self.EGeV)**self.pm_b['c21']
    def _set_a2b(self):
        if self.log10E >= self.pm_b['lb']:
            self.a2b = self.pm_b['a20'] * (self.EGeV)**self.pm_b['a21'] + \
            self.pm_b['a22']
        else:
            num = self.pm_b['a20'] * 10.**(self.pm_b['a21'] * self.pm_b['lb']) + \
            self.pm_b['a22'] - self.pm_b['a24']
            den = 10.**(self.pm_b['a23'] * self.pm_b['lb'])
            self.a2b = (num / den) * (self.EGeV)**self.pm_b['a23'] + self.pm_b['a24']
    def _set_theta_0b(self):
        if self.log10E >= self.pm_b['lc']:
            self.theta_0b = self.pm_b['p0'] * (self.EGeV)**self.pm_b['p1']
        else:
            self.theta_0b = self.pm_b['p0'] * 10**(self.pm_b['lc']*(self.pm_b['p1']
            - self.pm_b['p2'])) * (self.EGeV)**self.pm_b['p2']
    def _set_rb(self):
        self.rb = self.pm_b['r0']
        ld = self.pm_b['r2'] / self.pm_b['r3']
        if self.log10E <= ld:
            self.rb += self.pm_b['r1'] * (self.EGeV)**(self.pm_b['r2'] +
            self.pm_b['r3'] * self.log10E)

    def set_lE(self,lE):
        self.lE = lE #natural log of E in MeV
        self.E = np.exp(lE) #E in MeV
        self.EGeV = self.E * 1.e-3 #E in GeV
        self.log10E = np.log10(self.EGeV) #commonlog of E in GeV
        self.normalize()

    def set_schema(self,schema):
        """
        Reset schema and normalize
        """
        self.schema = schema
        self.normalize()

    def norm_integrand(self,theta):
        return self.n_t_lE_Omega(theta) * np.sin(theta) * 4 * np.pi

    def normalize(self):
        """Set the normalization constant so that the integral over radians is unity."""
        self.C0 =1
        if self.schema == 'b':
            self._set_a1b()
            self._set_c1b()
            self._set_c2b()
            self._set_a2b()
            self._set_theta_0b()
            self._set_rb()
        elif self.schema == 'l':
            self._set_b1l()
            self._set_b2l()
            self._set_a1l()
            self._set_a2l()
            self._set_sigl()
        intgrl = 0.
        for ll,ul in zip(self.lls,self.uls):
            intgrl += quad(self.norm_integrand,ll,ul)[0]
        self.C0 = 1/intgrl

    def n_t_lE_Omega(self,theta):
        """
        This function returns the particle angular distribution as a angle at a given energy.
        It is independent of particle type and shower stage
        Parameters:
            theta: the angle [rad]
        Returns:
            n_t_lE_Omega = the angular distribution of particles
        """

        dist_value = np.empty(1)
        if self.schema == 'b':
            if self.log10E > 3.: # if the energy is greater than 1 TeV return a narrow Gaussian
                sig = 5.e-4 * (1000./self.EGeV)
                dist_value = self.C0 * np.exp(-(theta**2)/(2*sig**2))
            else:
                t1 = self.a1b * np.exp(-self.c1b * theta - self.c2b * theta**2)
                t2 = self.a2b / ((1 + theta * self.theta_0b)**(self.rb))
                dist_value = self.C0 * (t1 + t2)
        elif self.schema =='l':
            theta = np.degrees(theta)
            t1 = np.exp(self.b1l) * theta**self.a1l
            t2 = np.exp(self.b2l) * theta**self.a2l
            mrs = -1/self.sigl
            ms = -self.sigl
            dist_value = self.C0 * (t1**mrs + t2**mrs)**ms
        return dist_value

class LateralDistribution:
    """
    This class contains functions related to the lateral distribution
    of secondary particles.  The parameterization used is that of
    S. Lafebre et. al. (2009).
    """
    # pm for parameter

    pm = {'xp11':0,'xp12':1,'xp13':2,'zp01':3,'zp02':4,'zp03':5,'zp04':6,'zp05':7,'zp11':8,'zp12':9}
    # pz for parameterization
    #             xp11   xp12    xp13    zp01   zp02 zp03  zp04   zp05    zp11   zp12
    pz = np.array([0.859,-0.0461,0.00428,0.0263,1.34,0.160,-0.0404,0.00276,0.0263,-4.33])

    ll = 1.e-3 #lower limit
    ul = 100 #upper limitl

    def __init__(self,lE,t):
        """Set the parameterization constants for this type (log)energy. The
        lateral distribution depends on the log energy and stage.

        Parameters:
            lE = The log of the energy (in MeV) at which the lateral
                 distribution is calculated
            t = shower stage
        """
        self.lE = lE
        self.t = t
        self.C0 = 1.
        self.normalize(t)

    def _set_xp1(self):
        xp11 = self.pz[self.pm['xp11']]
        xp12 = self.pz[self.pm['xp12']]
        xp13 = self.pz[self.pm['xp13']]
        self.xp1 = xp11 + xp12*self.lE**2 + xp13*self.lE**3

    def _set_zp0(self,t):
        zp01 = self.pz[self.pm['zp01']]
        zp02 = self.pz[self.pm['zp02']]
        zp03 = self.pz[self.pm['zp03']]
        zp04 = self.pz[self.pm['zp04']]
        zp05 = self.pz[self.pm['zp05']]
        self.zp0 = zp01*t + zp02 + zp03*self.lE + zp04*self.lE**2 + zp05*self.lE**3

    def _set_zp1(self,t):
        zp11 = self.pz[self.pm['zp11']]
        zp12 = self.pz[self.pm['zp12']]
        self.zp1 = zp11*t + zp12

    def normalize(self,t):
        self.C0 = 1.
        self._set_xp1()
        self._set_zp0(t)
        self._set_zp1(t)
        intgrl,eps = quad(self.n_t_lE_lX,self.ll,self.ul)
        self.C0 = 1/intgrl

    def set_lE(self,lE,t):
        self.lE = lE
        self.t = t
        self.normalize(t)

    def n_t_lE_lX(self,X):
        """
        This function returns the particle lateral distribution as an
        angle at a given energy.

        Parameters:
            X: dimensionless Moliere units

        Returns:
            n_t_lE_lX = the lateral distribution
        """

        return self.C0 * X**self.zp0 * (self.xp1 + X)**self.zp1

class LateralDistributionNKG:
    '''
    This class implements the energy independent lateral distribution
    parameterization.

    Parameters:

    t = shower stages
    '''

    pm = {'zp00':0,'zp01':1,'zp10':2,'zp11':3,'xp10':4}
    pz = np.array([0.0238,1.069,0.0238,2.918,0.430])
    ll = np.log(1.e-3)
    ul = np.log(1.e1)

    def __init__(self,t):
        self.t = t
        self.normalize(t)

    def set_zp0(self,t):
        zp00 = self.pz[self.pm['zp00']]
        zp01 = self.pz[self.pm['zp01']]
        self.zp0 = zp00 * t + zp01

    def set_zp1(self,t):
        zp10 = self.pz[self.pm['zp10']]
        zp11 = self.pz[self.pm['zp11']]
        self.zp1 = zp10 * t - zp11

    def set_xp1(self):
        self.xp1 = self.pz[self.pm['xp10']]

    def n_t_lX_of_X(self, X):
        """
        This function returns the particle lateral distribution as a
        function of the Moliere radius.

        Parameters:
        X = Moliere radius (dimensionless)

        Returns:
        n_t_lX = the normalized lateral distribution value at X
        """
        return self.C0 * X ** self.zp0 * (self.xp1 + X) ** self.zp1

    def n_t_lX(self,lX):
        """
        This function returns the particle lateral distribution as a
        function of the Moliere radius.

        Parameters:
        X = Moliere radius (dimensionless)

        Returns:
        n_t_lX = the normalized lateral distribution value at X
        """
        X = np.exp(lX)
        return self.n_t_lX_of_X(X)

    def N_t_lX(self,lX):
        return self.n_t_lX(lX) / np.exp(lX)#**2

    def set_t(self,t):
        self.t = t
        self.normalize(t)

    def normalize(self,t):
        self.C0 = 1.
        self.set_zp0(t)
        self.set_zp1(t)
        self.set_xp1()
        intgrl,eps = quad(self.n_t_lX,self.ll,self.ul)
        self.C0 = 1/intgrl
        self.AVG = self.AVG_Moliere()

    def AVG_integrand(self,X):
        return X * self.n_t_lX(X)

    def AVG_Moliere(self):
        intgrl,eps = quad(self.AVG_integrand,self.ll,self.ul)
        return intgrl

    def d_rho_dA_of_r_d(self, r: float, d: float) -> float:
        '''This method returns the fractional particle density at a distance r
        from a piont in the shower core.

        Parameters:
        r: distance from shower core (m)
        r_M: the approximate Moliere radius at the shower core (m)
        d: the atmospheric density at the core (kg / m^3)

        returns:
        fractional particle density (fraction of total particles / m^2)
        '''
        rM = self.moliere_radius(d)
        X = r / rM
        return self.n_t_lX_of_X(X) / (2. * np.pi * X**2 * rM**2)

    @staticmethod
    def moliere_radius(d: float) -> float:
        '''Returns approximate Moliere radius at density (d)

        Parameters:
        d: density (kg/m^3)
        returns:
        Moliere radius (m)
        '''
        return 96. / d

    def make_table(self):
        atm = Atmosphere()
        ts = np.linspace(-20.,20.,100)
        ds = np.linspace(atm.density(0.), atm.density(atm.maximum_height), 1000)
        rs = np.logspace(-3,3,300)
        drho_dN_of_t_d_r = np.empty((ts.size,ds.size,rs.size),dtype=float)
        for i, t in enumerate(ts):
            self.set_t(t)
            for j, d in enumerate(ds):
                drho_dN_of_t_d_r[i,j] = self.d_rho_dA_of_r_d(rs, d)
        return ts, ds, rs, drho_dN_of_t_d_r

    def make_lX_table(self):
        ts = np.linspace(-20.,20.,1000)
        # lXs = np.array([-3.5,-2.5,-1.5,-.5,.5])
        # lXs = np.arange(-4,1)
        lX_intervals = np.linspace(-6,1,15)
        lXs = (lX_intervals + (np.diff(lX_intervals)/2)[0])[:-1]
        n_t_lX_of_t_lX = np.empty((ts.size,lXs.size),dtype=float)
        for i, t in enumerate(ts):
            self.set_t(t)
            n_t_lX_of_t_lX[i] = self.N_t_lX(lXs) / self.N_t_lX(lXs).sum()
        np.savez('n_t_lX_of_t_lX.npz',n_t_lX_of_t_lX=n_t_lX_of_t_lX,ts=ts,lXs=lXs)


if __name__ == '__main__':
    ld = LateralDistributionNKG(0)
    ld.make_lX_table()
    # import matplotlib.pyplot as plt
    # plt.ion()
    # atm = Atmosphere()
    # d = atm.density(0.)
    #
    # x = np.logspace(-3,2,100)
    # lx = np.log(x)
    # ld = LateralDistributionNKG(0)
    # plt.figure()
    # plt.plot(x, ld.n_t_lX(lx), label = f"t = {ld.t}")
    # ld.set_t(-10)
    # plt.plot(x, ld.n_t_lX(lx), label = f"t = {ld.t}")
    # ld.set_t(10)
    # plt.plot(x, ld.n_t_lX(lx), label = f"t = {ld.t}")
    # plt.legend()
    # plt.loglog()
    # plt.xlabel('X (Moliere Units)')
    # plt.ylabel('n_t_lX')
    #
    # r = np.logspace(-3,3,100)
    # plt.figure()
    # ld.set_t(0.)
    # plt.plot(r, ld.d_rho_dA_of_r_d(r, d), label = f"t = {ld.t}")
    # ld.set_t(-10.)
    # plt.plot(r, ld.d_rho_dA_of_r_d(r, d), label = f"t = {ld.t}")
    # ld.set_t(10.)
    # plt.plot(r, ld.d_rho_dA_of_r_d(r, d), label = f"t = {ld.t}")
    # plt.legend()
    # plt.xlabel('distance from core (m)')
    # plt.ylabel('density (particles/m^2)')

    # ll = np.radians(0.1)
    # ul = np.radians(45.)
    # lqrad = np.linspace(np.log(ll),np.log(ul),450)
    # qrad = np.exp(lqrad)
    #
    # fig = plt.figure()
    # qd = AngularDistribution(np.log(1.),'l')
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='1 MeV')
    # qd.set_lE(np.log(5.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='5 MeV')
    # qd.set_lE(np.log(30.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='30 MeV')
    # qd.set_lE(np.log(170.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='170 MeV')
    # qd.set_lE(np.log(1.e3))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='1 GeV')
    # plt.loglog()
    # plt.legend()
    # plt.xlabel('Theta [rad]')
    # plt.ylabel('n(t;lE,Omega)')
    # plt.show()
    #
    # fig = plt.figure()
    # qd.set_schema('b')
    # qd.set_lE(np.log(1.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='1 MeV B')
    # qd.set_lE(np.log(5.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='5 MeV B')
    # qd.set_lE(np.log(30.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='30 MeV B')
    # qd.set_lE(np.log(170.))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='170 MeV B')
    # qd.set_lE(np.log(1.e3))
    # plt.plot(qrad,qd.n_t_lE_Omega(qrad),label='1 GeV B')
    # plt.loglog()
    # plt.xlim(ll,ul)
    # plt.legend()
    # plt.xlabel('Theta [rad]')
    # plt.ylabel('n(t;lE,Omega)')
