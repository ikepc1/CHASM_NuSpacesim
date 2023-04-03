import numpy as np
# from dataclasses import dataclass, field, fields
# from datetime import datetime
# import struct

from .eventio_types import *
from .simulation import *
from .block_data import *

SYNC_MARKER = b'\x37\x8a\x1f\xd4'
IACT_TYPES = {
    'RunHeader':1200,
    'InputCard':1212,
    'AtmosphericProfile':1216,
    'TelescopeDefinition':1201,
    'EventHeader':1202,
    'ArrayOffsets':1203,
    'Longitudinal':1211,
    'TelescopeData':1204,
    'Photons':1205,
    'CameraLayout':1206,
    'TriggerTime':1207,
    'PhotoElectrons':1208,
    'EventEnd':1209,
    'RunEnd':1210,   
}

IACT_OBJECTS = {
    'RunHeader': RunHeaderData,
    # 'InputCard': InputCardData,
    # 'AtmosphericProfile': AtmosphericProfileData,
    # 'TelescopeDefinition': TelescopeDefinitionData,
    # 'EventHeader': EventHeaderData,
    # 'ArrayOffsets': ArrayOffsetsData,
    # 'Longitudinal': LongitudinalData,
    # 'TelescopeData': TelescopeData,
    # 'Photons': PhotonsData,
    # 'CameraLayout': CameraLayoutData,
    # 'TriggerTime': TriggerTimeData,
    # 'PhotoElectrons': PhotoElectrons,
    # 'EventEnd': EventEnd,
    # 'RunEnd': RunEnd,   
}

def object_header_bytes(type: int, length: int, id: int = 0) -> bytearray:
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
    # type_bytes = int_to_bytes(type)
    # length_bytes = int_to_bytes(length)
    # return SYNC_MARKER + type_bytes + b'\x00\x00\x00\x00' + length_bytes
    return bytearray(
        SYNC_MARKER + Int(type).to_bytes()+ Int(id).to_bytes() + Int(length).to_bytes()
        )

class ImproperBlockSize(Exception):
    pass

def eventio_bytes() -> bytearray:
    '''This function returns the byte buffer of a mocked eventio file.
    '''
    byte_buffer = bytearray()
    for block_type in IACT_OBJECTS:
        block_bytes = block_to_bytes(IACT_OBJECTS[block_type]())
        byte_buffer.extend(object_header_bytes(IACT_TYPES[block_type], len(block_bytes)))
        byte_buffer.extend(block_bytes)
    return byte_buffer

if __name__ == '__main__':
    ev_bytes = eventio_bytes()