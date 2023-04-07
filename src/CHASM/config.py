from dataclasses import dataclass

from .atmosphere import Atmosphere, USStandardAtmosphere, CorsikaAtmosphere

@dataclass
class AxisConfig:
    '''This is the container for axis config parameters'''
    N_POINTS: int = 1000
    ATM: Atmosphere = CorsikaAtmosphere()
    # ATM: Atmosphere = USStandardAtmosphere()
