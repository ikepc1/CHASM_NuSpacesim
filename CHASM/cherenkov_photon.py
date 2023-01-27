import numpy as np
from scipy.integrate import quad
import scipy.constants as spc

from .atmosphere import *
from .charged_particle import EnergyDistribution, AngularDistribution

twopi = 2.*np.pi
m_e = spc.value('electron mass energy equivalent in MeV')

class CherenkovPhoton:
    inner_precision = 1.e-5
    outer_precision = 1.e-4

    def __init__(self,t,delta,ntheta=321,minlgtheta=-3.,maxlgtheta=0.2):
        """Create a normalized Cherenkov-photon angular distribution at the
        given shower stage, t, and with the index-of-refraction of the atmosphere
        given by delta. By default there are 321 logarithmically spaced angular
        points from 1 mR to pi/2 R.

        Parameters:
            t: shower stage
            delta: difference of index-of-refraction from unity
            ntheta: number of log-spaced theta points to create
            minlgtheta: the minimum value of lgtheta
            maxlgtheta: the maximum value of lgtheta
        """
        self.t = t
        self.delta = delta
        lgtheta,dlgtheta = np.linspace(minlgtheta,maxlgtheta,ntheta,retstep=True)
        self.theta = 10**lgtheta
        self.dtheta = 10**(lgtheta+dlgtheta/2) - 10**(lgtheta-dlgtheta/2)
        fe = EnergyDistribution('Tot',self.t)
        gg = np.array([CherenkovPhoton.outer_integral(q,fe,delta) for q in self.theta])
        # gg /= (gg*self.dtheta).sum()
        self.gg = gg

    def __str__(self):
        return "CherenkovPhoton Angular Distribution at t=%.1f and delta=%.2e"%(self.t,self.delta)

    def __repr__(self):
        return "ng_t_delta_Omega(%.1f,%.2e,%d,%.2e,%.2e)"%(
            self.t,self.delta,len(self.theta),self.theta[0],self.theta[-1])

    def ng_t_delta_Omega(self,theta):
        """Return the Cherenkov photon angular distribution at the given theta

        Parameters:
            theta: the angle at which to give the value of the angular distribution [rad]

        Returns:
            gg: the value of the angular distribution

        This function interpolates between the values of the precomputed table
        to give the desired value
        """
        i = np.searchsorted(self.theta,theta,side='right')
        if type(i) is np.int64:
            i = np.array([i])
            theta = np.array([theta])
            nin = 0
        else:
            nin = len(theta)

        j = i-1
        lf = j<0
        rg = i>=len(self.theta)
        ok = (~lf)*(~rg)
        s = np.empty_like(theta)
        s[ok] = np.log(theta[ok]/self.theta[j[ok]]) / np.log(self.theta[i[ok]]/self.theta[j[ok]])
        s[lf] = np.log(theta[lf]/self.theta[0])     / np.log(self.theta[1]/self.theta[0])
        s[rg] = np.log(theta[rg]/self.theta[-2])    / np.log(self.theta[-1]/self.theta[-2])
        y1 = np.empty_like(theta)
        y2 = np.empty_like(theta)
        y1[ok] = self.gg[j[ok]]
        y2[ok] = self.gg[i[ok]]
        y1[lf] = self.gg[0]
        y2[lf] = self.gg[1]
        y1[rg] = self.gg[-2]
        y2[rg] = self.gg[-1]

        y = y1*(y2/y1)**s
        return y if nin>0 else y[0]

    @staticmethod
    def spherical_cosines(A,B,c):
        """Return the angle C in spherical geometry where ABC is a spherical
        triangle, and c is the interior angle across from c.
        """
        return np.arccos( np.cos(A)*np.cos(B) + np.sin(A)*np.sin(B)*np.cos(c) )

    @staticmethod
    def cherenkov_threshold(delta):
        """Calculate the Cherenkov threshold for this atmosphere

        Parameters:
            delta: the index-of-refraction minus one (n-1)

        Returns:
            E_Ck: The Cherenkov threshold energy [MeV]
        """
        n     = 1 + delta
        beta  = 1/n
        gamma = 1/np.sqrt((1-beta**2))
        E_Ck = gamma*m_e
        return E_Ck

    @staticmethod
    def cherenkov_angle(E,delta):
        """Calculate the Cherenkov angle for a given log energy and atmosphere

        Parameters:
            E: The energy for the producing electron [MeV]
            delta: the index-of-refraction minus one (n-1)

        Returns:
            theta_g: The angle of the Cherenkov cone for this atmosphere
        """
        n = 1+delta
        gamma = E/m_e
        beta  = np.sqrt(1-1/gamma**2)
        rnbeta = 1/n/beta
        if type(rnbeta) is np.ndarray:
            theta_g = np.empty_like(rnbeta)
            theta_g[rnbeta<=1] = np.arccos(rnbeta[rnbeta<=1])
            theta_g[rnbeta>1] = 0
        else:
            theta_g = 0. if rnbeta>1 else np.arccos(rnbeta)
        return theta_g

    @staticmethod
    def cherenkov_yield(E,delta):
        """Calculate the relative Cherenkov efficiency at this electron
        energy

        Parameters:
            E: The energy for the producing electron [MeV]
            delta: the index-of-refraction minus one (n-1)

        Returns:
            Y_c: The relative Cherenkov efficiency
        """
        Y_c = 1 - (CherenkovPhoton.cherenkov_threshold(delta)/E)**2
        if type(Y_c) is np.ndarray:
            Y_c[Y_c<0] = 0
        else:
            Y_c = max(Y_c,0.)
        return Y_c

    @staticmethod
    def inner_integrand(phi_e,theta,theta_g,g_e):
        """The function returns the inner integrand of the Cherenkov photon
        angular distribution.

        Parameters:
            phi_e: the internal angle between the shower-photon plane and the
              electron-photon plane (this is the integration vairable)
            theta: the angle between the shower axis and the Cherenkov photon
            theta_g: the angle between the electron and the photon (the Cherenkov
              cone angle)
            g_e: an AngularDistribution object (normalized for the energy
              of theta_g!)
        Returns:
            the inner integrand
        """
        theta_e = CherenkovPhoton.spherical_cosines(theta,theta_g,phi_e)
        value = g_e.n_t_lE_Omega(theta_e/spc.degree) /spc.degree
        return value

    @staticmethod
    def inner_integral(theta,theta_g,g_e):
        """The function returns the inner integral of the Cherenkov photon
        angular distribution.

        Parameters:
            theta: the angle between the shower axis and the Cherenkov photon
            theta_g: the angle between the electron and the photon (the Cherenkov
              cone angle)
            g_e: an AngularDistribution object (normalized for the energy
              of theta_g!)
        Returns:
            the inner integral

        The inner integrand depends only on theta_e which comes from phi_e and
        sperical cosines, which takes the cos(phi_e). Thus the integral is symmetric
        about phi_e = 0, and we can do the half integral [0,pi] and multiply by 2.
        (Rather than the full integral [-pi,pi] or [0,2pi].)
        """
        theta_e_is_0 = ( CherenkovPhoton.spherical_cosines(theta,theta_g,0.) == 0. )
        inner_ll = 1.e-4 if theta_e_is_0 else 0.
        return 2.*quad( CherenkovPhoton.inner_integrand,inner_ll,np.pi,args=(theta,theta_g,g_e),
                        epsrel=CherenkovPhoton.inner_precision,
                        epsabs=CherenkovPhoton.inner_precision )[0]

    @staticmethod
    def outer_integrand(l_g,theta,f_e,delta):
        """The function returns the outer integrand of the Cherenkov photon
        angular distribution.

        Parameters:
            l_g: the logarithm of the energy for which the Cherenkov angle is
              whatever it is (this is the integration variable)
            theta: the angle between the shower axis and the Cherenkov photon
            f_e: an EnergyDistribution object (normalized!)
            delta: the index-of-refraction minus one (n-1)

        Returns:
            the outer integrand
        """
        E_g = np.exp(l_g)
        theta_g = CherenkovPhoton.cherenkov_angle(E_g,delta)
        cherenkov_yield = CherenkovPhoton.cherenkov_yield(E_g,delta)
        g_e = AngularDistribution(l_g)
        inner = CherenkovPhoton.inner_integral(theta,theta_g,g_e)
        value = np.sin(theta_g) * cherenkov_yield * f_e.spectrum(l_g) * inner
        return value

    @staticmethod
    def outer_integral(theta,f_e,delta):
        """The function returns the outer integral of the Cherenkov photon
        angular distribution.

        Parameters:
            theta: the angle between the shower axis and the Cherenkov photon
            f_e: an EnergyDistribution object (normalized!)
            t: the shower stage
            delta: the index-of-refraction minus one (n-1) for the Cherenkov angle
        Returns:
            the outer integral
        """
        ll = np.log(CherenkovPhoton.cherenkov_threshold(delta))
        ul = 13.8 # np.log(1.e6)
        return quad( CherenkovPhoton.outer_integrand,ll,ul,args=(theta,f_e,delta),
                     epsrel = CherenkovPhoton.outer_precision )[0]

def make_CherenkovPhoton_list(t,n_delta=176,min_lg_delta=-7,max_lg_delta=-3.5,
                              ntheta=321,minlgtheta=-3.,maxlgtheta=0.2):
    """Make an list of CherenkovPhoton distributions, all with the same stage
    but with a logrithmic array of delta values

    Parameters:
        t: the shower stage
        n_delta: number of point to sample a different delta
        min_lg_delta: the log10 of the minimum value of delta
        max_lg_delta: the log10 of the maximum value of delta

    Returns:
        gg_list: A list of CherenkovPhoton objects
    """
    delta = np.logspace(min_lg_delta,max_lg_delta,n_delta)
    gg_list = []
    for i,d in enumerate(delta):
        print("%2d %.2e"%(i,d))
        gg_list.append(CherenkovPhoton(t,d,
                                       ntheta=ntheta,
                                       minlgtheta=minlgtheta,
                                       maxlgtheta=maxlgtheta))
    return gg_list

def make_CherenkovPhoton_array(n_t=21,min_t=-20.,max_t=20.,
                               n_delta=176,min_lg_delta=-7,max_lg_delta=-3.5,
                               n_theta=321,min_lg_theta=-3,max_lg_theta=0.2):
    """Make an array of CherenkovPhoton distributions, with a linear array
    of shower stages and a logrithmic array of delta values

    Parameters:
        n_t: the number of shower stages
        min_t: the minimum shower stage
        max_t: the maximum shower stage
        n_delta: number of point to sample a different delta
        min_lg_delta: the log10 of the minimum value of delta
        max_lg_delta: the log10 of the maximum value of delta

    Returns:
        gg_array: A rank-3 numpy array of Cherenkov angular distribution values
            with a shape (n_t, n_delta, n_theta)
        t_array: A numpy array containing the stages used in gg_array
        delta_array: A numpy array containing the atmospheric delta values used
            in the gg_array
        theta_array: A numpy array containing the angle values used in gg_array

    This routine is impractical to actually use because it would take many days
    to complete. It is intended to show the stucture of the completed gg_array.
    """
    t_array = np.linspace(min_t,max_t,n_t)
    delta_array = np.logspace(min_lg_delta,max_lg_delta,n_delta)
    theta_array = np.logspace(min_lg_theta,max_lg_theta,n_theta)
    gg_array = np.empty((n_t,n_delta,n_theta),dtype=float)
    for i,t in enumerate(t_array):
        for j,d in enumerate(delta_array):
            print("%2d %.0f %2d %.2e"%(i,ti,j,d))
            gg = CherenkovPhoton(ti,d,
                                 ntheta=n_theta,
                                 minlgtheta=min_lg_theta,
                                 maxlgtheta=max_lg_theta)
            gg_array[i,j] = gg.gg
    return gg_array,t_array,delta_array,theta_array

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    plt.ion()

    start_time = time.time()
    gg_0_sl = CherenkovPhoton(0,3e-5)
    end_time = time.time()
    print("Generated one Cherenkov Photon angular distribution in %.1f s"%(
        end_time-start_time))

    fg = plt.figure(1,figsize=(6,8))
    a1 = fg.add_subplot(211)
    a1.plot(gg_0_sl.theta,gg_0_sl.gg)
    a1.loglog()
    a1.set_xlim(gg_0_sl.theta[0],gg_0_sl.theta[-1])
    ymn,ymx = a1.set_ylim()
    a1.set_ylim(1.e-5*ymx,ymx)
    a1.grid()
    a1.set_xlabel('Theta [rad]')
    a1.set_ylabel('ng_t_delta_Omega [1/sr]')
    a1.set_title(repr(gg_0_sl))
    a2 = fg.add_subplot(212)
    a2.plot(gg_0_sl.theta,gg_0_sl.gg)
    a2.set_xlim(0,0.1)
    a2.set_ylim(0,1.2*gg_0_sl.gg.max())
    a2.grid()
    a2.set_xlabel('Theta [rad]')
    a2.set_ylabel('ng_t_delta_Omega [1/sr]')
    a2.set_title(repr(gg_0_sl))
    fg.tight_layout()
