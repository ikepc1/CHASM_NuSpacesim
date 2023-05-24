import numpy as np
from dataclasses import dataclass, field, fields
from pathlib import Path
# from datetime import datetime
# import struct

from .simulation import ShowerSimulation
from .block_data import *
from .eventio_types import EventioType

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
    'InputCard': InputCardData,
    'AtmosphericProfile': AtmosphericProfileData,
    'TelescopeDefinition': TelescopeDefinitionData,
    'EventHeader': EventHeaderData,
    'ArrayOffsets': ArrayOffsetsData,
    'Longitudinal': LongitudinalData,
    'TelescopeData': TelescopeData,
    'Photons': PhotonsData,
    # 'CameraLayout': CameraLayoutData,
    # 'TriggerTime': TriggerTimeData,
    # 'PhotoElectrons': PhotoElectrons,
    'EventEnd': EventEnd,
    'RunEnd': RunEnd,   
}

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
    buffer.extend(eitype.tobytes())

def block_to_bytes(block_data: dataclass) -> bytearray:
    '''This function takes the values in a block data dataclass container and writes
    them to a bytearray.
    '''
    values =  value_list(block_data)
    byte_buffer = bytearray()
    for value in values:
        append_EIType_to_buffer(byte_buffer, value)
    return byte_buffer

def create_data_blocks(sig: ShowerSignal) -> dict[str, dataclass]:
    '''This function instantiates the data block containers.
    '''
    block_dict = {
    'RunHeader': RunHeaderData(),
    'InputCard': InputCardData(),
    'AtmosphericProfile': AtmosphericProfileData(),
    }
    block_dict['TelescopeDefinition'] = make_tel_def(sig)
    block_dict['EventHeader'] = make_event_header(sig)
    block_dict['ArrayOffsets'] = make_array_offsets(sig)
    block_dict['Longitudinal'] = make_longitudinal(sig)
    block_dict['TelescopeData'] = make_telescope_data(sig)
    block_dict['EventEnd'] = make_event_end(sig)
    block_dict['RunEnd'] = RunEnd()
    return block_dict

def set_subob_bit(length: int):
    '''This method toggles the 30th bit in the length int.
    '''
    mask = 1 << 30
    return(int(length) ^ mask)

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
    b = bytearray(
        SYNC_MARKER + Int(type).tobytes()+ Int(id).tobytes()# + Int(length).to_bytes()
        )
    
    '''
    If it's the TelescopeData block, the header needs to have the 'only sub-objects'
    flag set in the length word. 
    '''
    if type == 1204:
        length = set_subob_bit(length)

    b += Int(length).tobytes()
    return b

class ImproperBlockSize(Exception):
    pass

def eventio_bytes(sig: ShowerSignal) -> bytearray:
    '''This function returns the byte buffer of a mocked eventio file.
    '''
    byte_buffer = bytearray()
    data_blocks = create_data_blocks(sig)
    for block_type in data_blocks:
        block_bytes = block_to_bytes(data_blocks[block_type])
        length = len(block_bytes)
        byte_buffer.extend(object_header_bytes(IACT_TYPES[block_type], length))
        byte_buffer.extend(block_bytes)
    return byte_buffer

def write_ei_file(sig: ShowerSignal, filename: str) -> None:
    '''This function writes the CHASM output to an eventio file format.
    '''
    Path(filename).touch()
    with open(filename,'wb') as ei_file:
        ei_file.write(eventio_bytes(sig))
