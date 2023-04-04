from dataclasses import dataclass, field, fields
from datetime import datetime
import struct

from .eventio_types import *
from .config import AxisConfig

def value_list(block_data: dataclass) -> list[EventioType]:
    '''This property is the attributes in a block dataclass as a list in 
    the correct order.
    '''
    value_list = []
    for field in fields(block_data):
        attr = getattr(block_data, field.name)
        if type(attr) == list:
            value_list.extend(attr)
        else:
            value_list.append(attr)
    return value_list

def append_EIType_to_buffer(buffer: bytearray, eitype: EventioType) -> None:
    '''This function appends the Eventio style bytes representing the datatype to
    a bytearray.
    '''
    buffer.extend(eitype.to_bytes())

def block_to_bytes(block_data: dataclass) -> bytearray:
    '''This function takes the values in a block data dataclass container and writes
    them to a bytearray.
    '''
    values =  value_list(block_data)
    byte_buffer = bytearray()
    for value in values:
        append_EIType_to_buffer(byte_buffer, value)
    return byte_buffer

def date_to_float() -> Float:
    '''This function takes the current date and returns it as a float in YYMMDD
    format for the run header.
    '''
    return Float(float(datetime.now().strftime("%y%m%d")))

def consts_and_int_flags() -> list[Float]:
    '''This function returns the list of constants and interaction flags for words
    24-74 in the CORSIKA run header. They're copied from a corsika run using my build
    '''
    values = [ 6.3713152e+08, 6.0000000e+05,  2.0000000e+06,  0.0000000e+00,  \
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
    return [Float(v) for v in values]

def CKA() -> list[Float]:
    '''This funtion returns a list which fills the CKA field in the run header.
    '''
    values = [0.0000000e+00, 1.0000000e-01, 0.0000000e+00, 0.0000000e+00, \
            0.0000000e+00, 0.0000000e+00, 2.5000000e-01, 5.0000000e-01, \
            7.5000000e-01, 1.0000000e+00, 5.0000000e-01, 2.0000000e-01, \
            9.2640668e-01, 1.1240715e+00, 1.4960001e+02, 1.4960001e+02, \
            2.3553182e-01, 2.0600000e-01, 1.3500001e-01, 2.2200000e-01, \
            5.0000000e-01, 0.0000000e+00, 6.3559997e-01, 6.9276202e-01, \
            8.7419999e-01, 6.7830002e-01, 4.0689999e-01, 2.4432061e+00, \
            9.1240066e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, \
            0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 1.0000000e+05, \
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]
    return [Float(v) for v in values]

def CETA() -> list[Float]:
    '''This function returns a list which fills the CETA field in the run header.
    '''
    values = [3.972e-01, 7.265e-01, 9.575e-01, 9.000e-04, 2.070e+00]
    return [Float(v) for v in values]

def CSTRBA() -> list[Float]:
    '''This function returns a list which fills the CSTRBA field in the run header.
    '''
    values = [0.    , 0.    , 0.    , 0.    , 0.6409, 0.5163, 0.    , 0.\
            , 0.    , 0.678 , 0.914 ]
    return [Float(v) for v in values]

def HLAY() -> list[Float]:
    '''This function returns a list which fills the HLAY field in the run header.
    '''
    values = [0, 4.e5, 1.e6, 5.e6, 1.03e7]
    return [Float(v) for v in values]

def AATM() -> list[Float]:
    values = [-2.7132117e+02, -1.5779353e+02,  3.2117999e-01, -6.1936921e-04, \
            1.3547195e-03]
    return [Float(v) for v in values]

def BATM() -> list[Float]:
    values = [1.3045211e+03, 1.1858909e+03, 1.3132604e+03, 5.7174750e+02, 1.0000000e+00]
    return [Float(v) for v in values]

def CATM() -> list[Float]:
    values = [1.0921114e+06, 9.8753669e+05, 6.3614306e+05, 7.6491294e+05, 8.8579226e+09]
    return [Float(v) for v in values]

@dataclass
class RunHeaderData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    header.
    '''
    runh: Float = Float(struct.unpack('f',b'RUNH')[0])
    run_no: Float = Float(1.) #Run number
    date: Float = field(default_factory=date_to_float) #Date as float YYMMDD.
    version: Float = Float(7.741) #mock CORSIKA version
    n_obs: Float = Float(1.) #Number of observation levels
    obs_levels: list[Float] = field(default_factory= lambda: [Float(0.)] * 10) #observation levels
    e_slope: Float = Float(-1.) #-1.
    e_low: Float = Float(1.e8) #Lower limit of energy range (primary)
    e_high: Float = Float(1.e8) #Upper limit of energy range (primary)
    egs4_flag: Float = Float(1.) #egs4 flag placeholder
    nkg_flag: Float = Float(1.) #nkg flag placeholder
    had_ecutoff: Float = Float(.3) #hadron energy cutoff placeholder
    muon_ecutoff: Float = Float(.3) #muon energy cutoff placeholder
    e_ecutoff: Float = Float(.001) #electron energy cutoff placeholder
    p_ecutoff: Float = Float(.001) #photon energy cutoff placeholder
    consts_int_flags: list[Float] = field(default_factory=consts_and_int_flags)
    incl_obs_plane: list[Float] = field(default_factory= lambda: [Float(0.)] * 5)
    not_used_word79_92: list[Float] = field(default_factory= lambda: [Float(0.)] * 13)
    n_showers: Float = Float(1.)
    not_used_word94: Float = Float(0.)
    cka: list[Float] = field(default_factory=CKA)
    ceta: list[Float] = field(default_factory=CETA)
    cstrba: list[Float] = field(default_factory=CSTRBA)
    not_used_word151_247: list[Float] = field(default_factory=lambda: [Float(0.)] * 97)
    xscatt: Float = Float(0.)
    yscatt: Float = Float(0.)
    hlay: list[Float] = field(default_factory=HLAY)
    aatm: list[Float] = field(default_factory=AATM)
    batm: list[Float] = field(default_factory=BATM)
    catm: list[Float] = field(default_factory=CATM)
    nflain: Float = Float(0.)
    nfldif: Float = Float(0.)
    nflpi0_100Xnflpif: Float = Float(0.)
    nflche_100Xnfragm: Float = Float(200.)

@dataclass
class InputCardData:
    '''This is a wrapper for the list of strings representing lines of 
    a CORSIKA steering file.
    '''
    lines: list[String] = field(default_factory= lambda: [String('Input Card Placeholder')])

def atm_altitudes_km() -> list[Double]:
    '''This gets the altitudes from the axis atmosphere config file'''
    return [Double(val / 1.e3) for val in AxisConfig().ATM.altitudes.tolist()]

def atm_table() -> list[Double]:
    '''This takes the atmosphere object from config and converts the 
    values into a list in the appropriate order for the eventio 
    block.
    '''
    atm = AxisConfig().ATM
    atm_array = np.empty((atm.altitudes.size,4))
    atm_array[:,0] = atm.altitudes / 1.e3 #to km
    atm_array[:,1] = atm.density(atm.altitudes) / 1.e3 #to g/cm^3
    atm_array[:,2] = atm.thickness()
    atm_array[:,3] = atm.delta(atm.altitudes)
    return [Double(val) for val in atm_array.flatten().tolist()]

def five_layer() -> list[Double]:
    pass

@dataclass
class AtmosphericProfileData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    atmospheric profile block.
    '''
    name: Varstring = Varstring(AxisConfig().ATM.name)
    obslev: Double = Double(0.)
    table_size: Varint = Varint(len(AxisConfig().ATM.altitudes.tolist()))
    altitude_table: list[Double] = field(default_factory=atm_table)
    n_five_layer: Varint = Varint(0)
    # htoa: Double = Double(AxisConfig().ATM.altitudes.max())
    # five_layer_atm: list[Double] = field(default_factory=five_layer)

@dataclass
class TelescopeDefinitionData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    IACT definition block.
    '''
    n_tel: Varint
    empty_word: Float


@dataclass
class EventHeaderData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    eventio event header block.
    '''

@dataclass
class ArrayOffsetsData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    array offsets block.
    '''

@dataclass
class LongitudinalData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    longitudinal block.
    '''

@dataclass
class TelescopeData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    telescope events block.
    '''

@dataclass
class PhotonsData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    photon bunch block.
    '''

@dataclass
class CameraLayoutData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    camera layout block.
    '''

@dataclass
class TriggerTimeData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    trigger time block.
    '''

@dataclass
class PhotoElectrons:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    photo electrons block.
    '''

@dataclass
class EventEnd:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    event end block.
    '''

@dataclass
class RunEnd:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    run end block.
    '''

# def run_header_bytes() -> bytearray:
#     bb = object_header_bytes(IACT_TYPES['RunHeader'], 1096)
#     bb.extend(block_to_bytes(RunHeaderData()))
#     return bb

# def input_card_bytes(card_lines: list[str]) -> bytearray:
#     '''This function returns the byte buffer representing the eventio input 
#     card data block'''
#     block = object_header_bytes(IACT_TYPES['InputCard'], len(block))
#     block.extend(len(card_lines).to_bytes(4,'little'))
#     for line in card_lines:
#         block.extend(String(line).to_bytes)
#     return block