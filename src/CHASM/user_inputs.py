from dataclasses import dataclass, field

from .axis import *
from .generate_Cherenkov import MakeYield
from .shower import *

@dataclass
class AxisParamContainer:
    '''This is the base class for axis params since both upward
    and downward axes have the same parameters.
    '''
    zenith: float
    azimuth: float
    ground_level: float = 0.
    curved: bool = False
    element_type: str = field(init=False, default='axis', repr=False)

@dataclass
class DownwardAxis(AxisParamContainer):
    '''This is the container for a user-defined downward axis' parameters.
    '''
    def create(self) -> MakeDownwardAxis:
        '''This method returns the instantiated axis object.
        '''
        if self.curved:
            return MakeDownwardAxisCurvedAtm(self)
        else:
            return MakeDownwardAxisFlatPlanarAtm(self)
    
@dataclass
class UpwardAxis(AxisParamContainer):
    '''This is the container for a user-defined upward axis' parameters.
    '''
    def create(self) -> MakeUpwardAxis:
        '''This method returns the instantiated axis object.
        '''
        if self.curved:
            return MakeUpwardAxisCurvedAtm(self)
        else:
            return MakeUpwardAxisFlatPlanarAtm(self)

@dataclass
class GHShower:
    '''This is the GH shower ingredient parameter container/factory'''
    X_max: float
    N_max: float
    X0: float
    Lambda: float
    element_type: str = field(init=False, default='shower', repr=False)

    def create(self) -> MakeGHShower:
        '''This method returns an instantiated Gaisser Hillas Shower '''
        return MakeGHShower(self.X_max, self.N_max, self.X0, self.Lambda)

@dataclass
class UserShower:
    '''This is the implementation of the GH shower element'''
    X: np.ndarray
    Nch: np.ndarray
    element_type: str = field(init=False, default='shower', repr=False)

    def create(self) -> MakeUserShower:
        '''This method returns an instantiated user shower '''
        return MakeUserShower(self.X, self.Nch)

@dataclass
class CountersParamsContainer:
    '''Different counter types take the same params'''
    vectors: np.ndarray
    radius: float
    element_type: str = field(init=False, default='counters', repr=False)

@dataclass
class SphericalCounters(CountersParamsContainer):
    '''This is the implementation of the spherical counters array.'''

    def create(self) -> MakeSphericalCounters:
        '''This method returns an instantiated spherical counter array'''
        return MakeSphericalCounters(self.vectors, self.radius)

@dataclass
class FlatCounters(CountersParamsContainer):
    '''This is the implementation of the flat counters array'''

    def create(self) -> MakeFlatCounters:
        '''This method returns an instantiated orbital array'''
        return MakeFlatCounters(self.vectors, self.radius)

@dataclass
class Yield:
    '''This is the implementation of the yield element'''
    l_min: float
    l_max: float
    N_bins: int = 10
    element_type: str = field(init=False, default='yield', repr=False)

    def make_lambda_bins(self):
        '''This method creates a list of bin low edges and a list of bin high
        edges'''
        bin_edges = np.linspace(self.l_min, self.l_max, self.N_bins+1)
        return bin_edges[:-1], bin_edges[1:]

    def create(self) -> list[MakeYield]:
        '''This method returns an instantiated yield object'''
        bin_minimums, bin_maximums = self.make_lambda_bins()
        yield_array = []
        for i, (min, max) in enumerate(zip(bin_minimums, bin_maximums)):
            yield_array.append(MakeYield(min, max))
        return yield_array