from dataclasses import dataclass

from .atmosphere import Atmosphere, USStandardAtmosphere, CorsikaAtmosphere

@dataclass
class AxisConfig:
    '''This is the container for axis config parameters.
    '''
    N_POINTS: int = 1000
    N_IN_RING: int = 5
    MIN_CHARGED_PARTICLES: float = 7.e4 #number of charged particles for a step to be considered in cherenkov calcs
    ATM: Atmosphere = CorsikaAtmosphere()
    # ATM: Atmosphere = USStandardAtmosphere()
  