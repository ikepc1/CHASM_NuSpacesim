import numpy as np
from dataclasses import dataclass, field, fields
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
    'CameraLayout': CameraLayoutData,
    'TriggerTime': TriggerTimeData,
    'PhotoElectrons': PhotoElectrons,
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

def create_data_blocks(sim: ShowerSimulation) -> dict[str, dataclass]:
    '''This function instantiates the data block containers.
    '''
    sig = sim.run()
    block_dict = {
    # 'RunHeader': RunHeaderData(),
    # 'InputCard': InputCardData(),
    # 'AtmosphericProfile': AtmosphericProfileData(),
    }
    # block_dict['TelescopeDefinition'] = make_tel_def(sim)
    # block_dict['EventHeader'] = make_event_header(sim)
    block_dict['Longitudinal'] = make_longitudinal(sig)
    return block_dict

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

def eventio_bytes(sim: ShowerSimulation) -> bytearray:
    '''This function returns the byte buffer of a mocked eventio file.
    '''
    byte_buffer = bytearray()
    data_blocks = create_data_blocks(sim)
    for block_type in data_blocks:
        block_bytes = block_to_bytes(data_blocks[block_type])
        byte_buffer.extend(object_header_bytes(IACT_TYPES[block_type], len(block_bytes)))
        byte_buffer.extend(block_bytes)
    return byte_buffer
