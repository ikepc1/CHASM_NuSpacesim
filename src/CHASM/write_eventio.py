import numpy as np
from dataclasses import dataclass, field, fields
from datetime import datetime
import struct

from .eventio_types import float_to_bytes, string_to_bytes
from .simulation import *

SYNC_MARKER = b'\x37\x8a\x1f\xd4'
IACT_TYPES = {
    'RunHeader':1200,
    'TelescopeDefinition':1201,
    'EventHeader':1202,
    'ArrayOffsets':1203,
    'TelescopeData':1204,
    'Photons':1205,
    'CameraLayout':1206,
    'TriggerTime':1207,
    'PhotoElectrons':1208,
    'EventEnd':1209,
    'RunEnd':1210,
    'Longitudinal':1211,
    'InputCard':1212,
    'AtmosphericProfile':1216,
}

def object_header_bytes(type: int, length: int) -> bytes:
    '''This function writes the header bytes for an eventio file. These 16 
    bytes correspond to:
    0:4 -> Synchronization marker (data bytes always reversed, see eventio doc)
    4:8 -> Type/version number
    8:12 -> Identification field
    12:16 -> Length field
    Parameters:
    type: int -> the iact type no
    length: int -> size of the data block in bytes
    returns: bytes -> the header bytes
    '''
    type_bytes = type.to_bytes(4,'little')
    length_bytes = length.to_bytes(4,'little')
    return SYNC_MARKER + type_bytes + b'\x00\x00\x00\x00' + length_bytes

class ImproperBlockSize(Exception):
    pass

class BlockData:
    '''This class contains the methods needed to convert the data fields in a 
    block dataclass to a bytestream.
    '''

    @property
    def value_list(self):
        '''This property is the attributes in a block dataclass as a list in 
        the correct order.
        '''
        value_list = []
        for field in fields(self):
            attr = getattr(self, field.name)
            if type(attr) == float:
                value_list.append(attr)
            elif type(attr) == list:
                value_list.extend(attr)
            else:
                pass
        return value_list

    def size_bytes(self):
        '''This method returns the number of 4 byte words in the data block.
        '''
        return len(self.value_list).to_bytes(4,'little')

    def to_bytes(self):
        '''This method concatenates the attributes of a block dataclass into an 
        eventio style byte buffer.
        '''
        byte_buffer = self.size_bytes()
        for value in self.value_list:
            byte_buffer += float_to_bytes(value)
        return byte_buffer

def date_to_float() -> float:
    '''This function takes the current date and returns it as a float in YYMMDD
    format for the run header.
    '''
    return float(datetime.now().strftime("%y%m%d"))

def consts_and_int_flags() -> list[float]:
    '''This function returns the list of constants and interaction flags for words
    24-74 in the CORSIKA run header. They're copied from a corsika run using my build
    '''
    return [ 6.3713152e+08, 6.0000000e+05,  2.0000000e+06,  0.0000000e+00,  \
             0.0000000e+00,  4.5805965e-02,  5.7308966e-01,  5.2830420e-02,  \
             2.5000000e+00,  2.0699999e+00,  8.1999998e+00,  1.0000000e-01,  \
             0.0000000e+00,  0.0000000e+00,  1.0000234e+00,  9.6726632e-03,  \
             1.0000001e+00,  5.7517074e-04,  0.0000000e+00,  0.0000000e+00,  \
             3.7700001e+01,  1.5328730e-04,  9.3864174e+00,  2.0000001e-03,  \
             2.9979247e+10,  1.0000000e+00,  5.4030228e-01,  3.1415927e+00,  \
             -1.0000000e+00,  2.1000000e-02,  0.0000000e+00,  0.0000000e+00, \
             0.0000000e+00,  2.0000000e+01,  0.0000000e+00,  0.0000000e+00,  \
             .0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,   \
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  \
             2.6190960e-01,  8.9983428e-01,  0.0000000e+00,  1.0389919e+00,  \
             2.7138337e-01,  1.3703600e+02]

def CKA() -> list[float]:
    '''This funtion returns a list which fills the CKA field in the run header.
    '''
    return [0.0000000e+00, 1.0000000e-01, 0.0000000e+00, 0.0000000e+00, \
            0.0000000e+00, 0.0000000e+00, 2.5000000e-01, 5.0000000e-01, \
            7.5000000e-01, 1.0000000e+00, 5.0000000e-01, 2.0000000e-01, \
            9.2640668e-01, 1.1240715e+00, 1.4960001e+02, 1.4960001e+02, \
            2.3553182e-01, 2.0600000e-01, 1.3500001e-01, 2.2200000e-01, \
            5.0000000e-01, 0.0000000e+00, 6.3559997e-01, 6.9276202e-01, \
            8.7419999e-01, 6.7830002e-01, 4.0689999e-01, 2.4432061e+00, \
            9.1240066e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, \
            0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 1.0000000e+05, \
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]

def CETA() -> list[float]:
    '''This function returns a list which fills the CETA field in the run header.
    '''
    return [3.972e-01, 7.265e-01, 9.575e-01, 9.000e-04, 2.070e+00]

def CSTRBA() -> list[float]:
    '''This function returns a list which fills the CSTRBA field in the run header.
    '''
    return [0.    , 0.    , 0.    , 0.    , 0.6409, 0.5163, 0.    , 0.\
            , 0.    , 0.678 , 0.914 ]

def HLAY() -> list[float]:
    '''This function returns a list which fills the HLAY field in the run header.
    '''
    return [0, 4.e5, 1.e6, 5.e6, 1.03e7]

def AATM() -> list[float]:
    return [-2.7132117e+02, -1.5779353e+02,  3.2117999e-01, -6.1936921e-04, \
            1.3547195e-03]

def BATM() -> list[float]:
    return [1.3045211e+03, 1.1858909e+03, 1.3132604e+03, 5.7174750e+02, 1.0000000e+00]

def CATM() -> list[float]:
    return [1.0921114e+06, 9.8753669e+05, 6.3614306e+05, 7.6491294e+05, 8.8579226e+09]

@dataclass
class RunHeaderData(BlockData):
    '''This class contains all the parameters needed to construct a mock CORSIKA
    header.
    '''
    runh: float = struct.unpack('f',b'RUNH')[0]
    run_no: float = 1. #Run number
    date: float = field(default_factory=date_to_float) #Date as float YYMMDD.
    version: float = 7.741 #mock CORSIKA version
    n_obs: float = 1. #Number of observation levels
    obs_levels: list[float] = field(default_factory= lambda: [0.] * 10) #observation levels
    e_slope: float = -1. #-1.
    e_low: float = 1.e8 #Lower limit of energy range (primary)
    e_high: float = 1.e8 #Upper limit of energy range (primary)
    egs4_flag: float = 1. #egs4 flag placeholder
    nkg_flag: float = 1. #nkg flag placeholder
    had_ecutoff: float = .3 #hadron energy cutoff placeholder
    muon_ecutoff: float = .3 #muon energy cutoff placeholder
    e_ecutoff: float = .001 #electron energy cutoff placeholder
    p_ecutoff: float = .001 #photon energy cutoff placeholder
    consts_int_flags: list[float] = field(default_factory=consts_and_int_flags)
    incl_obs_plane: list[float] = field(default_factory= lambda: [0.] * 5)
    not_used_word79_92: list[float] = field(default_factory= lambda: [0.] * 13)
    n_showers: float = 1.
    not_used_word94: float = 0.
    cka: list[float] = field(default_factory=CKA)
    ceta: list[float] = field(default_factory=CETA)
    cstrba: list[float] = field(default_factory=CSTRBA)
    not_used_word151_247: list[float] = field(default_factory=lambda: [0.] * 97)
    xscatt: float = 0.
    yscatt: float = 0.
    hlay: list[float] = field(default_factory=HLAY)
    aatm: list[float] = field(default_factory=AATM)
    batm: list[float] = field(default_factory=BATM)
    catm: list[float] = field(default_factory=CATM)
    nflain: float = 0.
    nfldif: float = 0.
    nflpi0_100Xnflpif: float = 0.
    nflche_100Xnfragm: float = 200.

def header_bytes() -> bytes:
    bb = object_header_bytes(IACT_TYPES['RunHeader'], 1096)
    bb += RunHeaderData().to_bytes()
    return bb

def input_card_bytes(card_lines: list[str]) -> bytes:
    '''This function returns the byte buffer representing the eventio input 
    card data block'''
    n_lines = len(card_lines)
    block = n_lines.to_bytes(4,'little')
    for line in card_lines:
        block += string_to_bytes(line)
    return object_header_bytes(IACT_TYPES['InputCard'], len(block)) + block



if __name__ == '__main__':
    main()