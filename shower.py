from abc import ABC, abstractmethod
import numpy as np

class Shower(ABC):
    '''This is the abstract class containing the needed methods for creating
    a shower profile
    '''
    @property
    def X_max(self):
        '''X_max getter'''
        return self._X_max

    @X_max.setter
    def X_max(self, X_max):
        '''X_max property setter'''
        if X_max <= 0.:
            raise ValueError("Negative X_max")
        self._X_max = X_max

    @property
    def N_max(self):
        '''N_max getter'''
        return self._N_max

    @N_max.setter
    def N_max(self, N_max):
        '''N_max property setter'''
        if not N_max.is_integer():
            raise ValueError("Non integer N_max")
        self._N_max = N_max

    @property
    def X0(self):
        '''X0 getter'''
        return self._X0

    @X0.setter
    def X0(self, X0):
        '''X0 property setter'''
        if X0 < 0.:
            raise ValueError("Negative X0")
        self._X0 = X0

    def stage(self, X, X0=36.62):
        '''returns stage as a function of shower depth'''
        return (X-self.X_max)/X0

    @abstractmethod
    def profile(self,*args,**kwargs):
        '''returns the number of charged particles as a function of depth'''

class MakeGHShower(Shower):
    '''This is the implementation where a shower profile is computed by the
    Gaisser-Hillas function'''

    def __init__(self, X_max: float, N_max: float, X0: float, Lambda: float):
        self.X_max = X_max
        self.N_max = N_max
        self.X0 = X0
        self.Lambda = Lambda

    def __repr__(self):
        return "GHShower(X_max={:.2f} g/cm^2, N_max={:.2f} particles, X0={:.2f} g/cm^2, Lambda={:.2f})".format(
        self.X_max, self.N_max, self.X0, self.Lambda)

    @property
    def Lambda(self):
        '''Lambda property getter'''
        return self._Lambda

    @Lambda.setter
    def Lambda(self, Lambda):
        '''Lambda property setter'''
        if Lambda <= 0.:
            raise ValueError("Negative Lambda")
        self._Lambda = Lambda

    def profile(self, X: np.ndarray):
        '''Return the size of a GH shower at a given depth.
        Parameters:
        X: depth

        Returns:
        # of charged particles
        '''
        x =         (X-self.X0)/self.Lambda
        g0 = x>0.
        m = (self.X_max-self.X0)/self.Lambda
        n = np.zeros_like(x)
        n[g0] = np.exp( m*(np.log(x[g0])-np.log(m)) - (x[g0]-m) )
        return self.N_max * n

class MakeUserShower(Shower):
    '''This is the implementation where a shower profile given by the user'''

    def __init__(self,X: np.ndarray, Nch: np.ndarray):
        if X.size != Nch.size:
            raise ValueError('Input arrays are not the same size')
        self.input_X = X
        self.input_Nch = Nch
        self.X_max = X[np.argmax(Nch)]
        self.N_max = np.rint( Nch.max() )
        self.X0 = X.min()

    def __repr__(self):
        return "UserShower(X_max={:.2f} g/cm^2, N_max={:.2f} particles, X0={:.2f} g/cm^2)".format(
        self.X_max, self.N_max, self.X0)

    @property
    def input_X(self):
        '''Input X getter'''
        return self._input_X

    @input_X.setter
    def input_X(self,X):
        '''Input X setter'''
        if type(X) != np.ndarray:
            X = np.array(X)
        self._input_X = X

    @property
    def input_Nch(self):
        '''Input Nch getter'''
        return self._input_Nch

    @input_Nch.setter
    def input_Nch(self,Nch):
        '''Input Nch setter'''
        if type(Nch) != np.ndarray:
            Nch = np.array(Nch)
        self._input_Nch = Nch


    def profile(self, X: np.ndarray):
        """Return the size of the shower at a slant-depth X

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]

        Returns:
            N: the shower size
        """
        return np.interp(X, self.input_X, self.input_Nch)
