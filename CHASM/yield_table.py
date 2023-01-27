import numpy as np
from scipy.integrate import quad
from scipy.constants import value,nano,mega

from .charged_particle import EnergyDistribution
from .cherenkov_photon import CherenkovPhoton as cp
from .cherenkov_photon_array import *

class GammaYield:
    '''This class contains the methods needed to calculate the average photon
    yield for all the charged particles in a shower at a given stage of shower
    development and atmospheric delta
    '''
    c = value('speed of light in vacuum')
    hc = value('Planck constant in eV s') * c
    me = value('electron mass')
    me_MeV = value('electron mass energy equivalent in MeV')
    alpha = value('fine-structure constant')
    ed = EnergyDistribution('Tot', 0.)

    def __init__(self, l_min = 300., l_max = 450.):
        self.l_min = l_min * nano
        self.l_max = l_max * nano

    @property
    def gammas_per_meter(self):
        '''This is the maximum number of Cherenkov photons produced per meter
        by an electron per wavelength interval (in the limit where beta >> 1/n).
        '''
        return 2 * np.pi * self.alpha # * (1 / self.l_min - 1 / self.l_max)

    def photon_yield(self, lE: np.ndarray, delta: np.ndarray) -> np.ndarray:
        '''This method computes the Cherenkov photon yield per meter of an
        electron with log energy le propagating through a medium with
        delta = n - 1. If le and delta are arrays they must be broadcastable.

        parameters:
        le = log(E (MeV)) single value or array
        delta = n -1 single value or array

        Returns:
        The # of Cherenkov photons / meter / charged particle
        '''
        E_e = np.exp(lE) #MeV
        n = delta + 1
        return self.gammas_per_meter * (1 - 1 / (n**2 * self.beta2(E_e)))

    @staticmethod
    def beta2(E_e: np.ndarray) -> np.ndarray:
        '''This method returns beta squared for an elecron of energy E
        Parameters:
        E_e = electron energy (MeV)

        Returns:
        Beta squared
        '''
        return 1 - (GammaYield.me_MeV / E_e)**2

    def avg_yield_integrand(self, lE: float, ed: EnergyDistribution, delta: float):
        '''This method is the integrand representing the weighted contibution
        of a specific energy interval
        '''
        return self.photon_yield(lE, delta) * ed.spectrum(lE)

    def avg_yield_integral(self, ed: EnergyDistribution, delta):
        '''This method is returns the average Cherenkov photon yield per charged
        particle at stage t and atmospheic delta = n - 1

        Parameters:
        ed = EnergyDistribution object normalized for the desired stage
        delta = atmospheric delta

        Returns:
        average Cherenkov photon yield per charged particle
        '''
        ll = np.log(cp.cherenkov_threshold(delta))
        ul = ed.ul
        return quad(self.avg_yield_integrand, ll, ul, args = (ed, delta))[0]

class YieldTable:
    cpa = CherenkovPhotonArray('gg_t_delta_theta_doubled.npz')
    delta = cpa.delta
    gy = GammaYield()
    ed = EnergyDistribution('Tot', 0)

    def __init__(self, N_t: int):
        self.t = np.linspace(-20. ,20, N_t)
        self.make_table()

    def make_table(self):
        self.y_t_delta = np.empty((self.t.size,self.delta.size))
        for i,t in enumerate(self.t):
            ed = EnergyDistribution('Tot', t)
            for j,d in enumerate(self.delta):
                self.y_t_delta[i,j] = self.gy.avg_yield_integral(ed,d)
        np.savez('y_t_delta.npz', y_t_delta = self.y_t_delta, t=self.t, d=self.delta)

if __name__ == 'main':
    yt = YieldTable(1000)
