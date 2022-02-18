import numpy as np
from counters import Counters
from axis import Axis
from shower import Shower
from scipy.constants import value,nano
from cherenkov_photon import CherenkovPhoton
from cherenkov_photon_array import CherenkovPhotonArray

class MakeYield:
    '''This class interacts with the table of Cherenkov yield ratios'''

    def __init__(self, l_min: float, l_max: float, npzfile = 'y_t_delta.npz'):
        y = np.load(npzfile)
        self.y_delta_t = y['y_t_delta'] * self.lambda_interval(l_min, l_max)
        self.delta = y['d']
        self.t = y['t']

    def lambda_interval(self, l_min, l_max):
        '''This method returns the factor that results from integrating the
        1/lambda^2 factor in the Frank Tamm formula
        '''
        return 1 / (l_min * nano) - 1 / (l_max * nano)

    def y_of_t(self, t: float):
        '''This method returns the array of yields (for each delta) of the
        tabulated stage nearest to the given t
        '''
        return self.y_delta_t[np.abs(t - self.t).argmin()]

    def y(self, d: float, t: float):
        '''This method returns the average Cherenkov photon yield per meter
        per charged particle at a given stage and delta.
        '''
        return np.interp(d, self.delta, self.y_of_t(t))

    def y_list(self, t_array: np.ndarray, delta_array: np.ndarray):
        '''This method returns a list of average Cherenkov photon yields
        corresponding to a list of stages and deltas.
        Parameters:
        t_array: numpy array of stages
        delta_array: numpy array of corresponding deltas

        Returns: numpy array of yields
        '''
        y_array = np.empty_like(t_array)
        for i,t in enumerate(t_array):
            y_array[i] = self.y(delta_array[i], t)
        return y_array
