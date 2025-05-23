from dataclasses import dataclass

from .atmosphere import Atmosphere, USStandardAtmosphere, CorsikaAtmosphere

@dataclass
class AxisConfig:
    '''This is the container for axis config parameters.
    '''
    N_POINTS: int = 1000
    N_IN_RING: int = 3
    MIN_CHARGED_PARTICLES: float = 1.e-4 #number of charged particles for a step to be considered in cherenkov calcs as a fraction of Nmax
    ATM: Atmosphere = CorsikaAtmosphere()
    # ATM: Atmosphere = USStandardAtmosphere()
    MAX_RING_SIZE: float = 300.
  