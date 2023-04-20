from dataclasses import dataclass

from .atmosphere import Atmosphere, USStandardAtmosphere, CorsikaAtmosphere

@dataclass
class AxisConfig:
    '''This is the container for axis config parameters'''
    N_POINTS: int = 500
    N_IN_RING: int = 20
    ATM: Atmosphere = CorsikaAtmosphere()
    # ATM: Atmosphere = USStandardAtmosphere()
