import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
import numpy as np

class EventioType(ABC):
    '''This is the base class for an eventio datatype.
    '''

    @abstractmethod
    def to_bytes(self) -> bytes:
        '''This method should return the eventio style bytes of the type 
        corresponding to the implementation.
        '''

@dataclass
class Float(EventioType):
    ''''This is the implementation of an eventio style float.
    '''
    value: float

    def to_bytes(self) -> bytes:
        return struct.pack('<f', self.value)

# def float_to_bytes(input_float: float) -> bytes:
#     '''This function returns the little endian 4 byte floats found in the eventio
#     bytestream.'''
#     return struct.pack('<f', input_float)

@dataclass
class String(EventioType):
    '''This is the implementation of an eventio style string with a 2 byte
    short integer representing the length prepended.
    '''
    value: str

    def to_bytes(self) -> bytes:
        byte_string = bytes(self.value, 'utf8')
        return struct.pack('<h',len(byte_string)) + byte_string

@dataclass
class Int(EventioType):
    ''''This is the implementation of the little endian 4 byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def to_bytes(self) -> bytes:
        return self.value.to_bytes(4,'little')
    
# def string_to_bytes(input_string: str) -> bytes:
#     '''This function returns an eventio style string byte buffer with the 
#     length prepended as a 2 byte unsigned short.
#     '''
#     byte_string = bytes(input_string, 'utf8')
#     return struct.pack('<h',len(byte_string)) + byte_string

# def int_to_bytes(input_int: int) -> bytes:
#     '''This function returns the little endian 4 byte ints found in the eventio
#     bytestream.
#     '''
#     return input_int.to_bytes(4,'little')

@dataclass
class Varint(EventioType):
    ''''This is the implementation of the single byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def to_bytes(self) -> bytes:
        return self.value.to_bytes(1, 'little')

@dataclass
class Double(EventioType):
    ''''This is the implementation of the single byte ints found in the 
    eventio bytestream.
    '''
    value: np.float64

    def to_bytes(self) -> bytes:
        return struct.pack('<d', self.value)

# def varint_to_bytes(input_int: int) -> bytes:
#     '''This function writes the single byte varint'''
#     return input_int.to_bytes(1, 'little')

# def double_to_bytes(input_float: float) -> bytes:
#     '''This function writes an eventio style double.'''
#     return struct.pack('<d', input_float)