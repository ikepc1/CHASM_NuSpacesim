import struct
from typing import Protocol
from dataclasses import dataclass
import numpy as np

class EventioType(Protocol):
    '''This is the base class for an eventio datatype.
    '''

    def to_bytes(self) -> bytes:
        '''This method should return the eventio style bytes of the type 
        corresponding to the implementation.
        '''

def array_to_eventio_list(arr: np.ndarray, type: EventioType) -> list[EventioType]:
    '''This method takes a numpy array arr (1d or 2d) and returns a list of
    the values converted to eventio types in the order needed for the 
    bytestream.
    '''
    return [type(val) for val in arr.flatten().tolist()]

@dataclass(frozen=True)
class Float:
    ''''This is the implementation of an eventio style float.
    '''
    value: float

    def to_bytes(self) -> bytes:
        return struct.pack('<f', self.value)

@dataclass(frozen=True)
class String:
    '''This is the implementation of an eventio style string with a 2 byte
    short integer representing the length prepended.
    '''
    value: str

    def to_bytes(self) -> bytes:
        byte_string = bytes(self.value, 'utf8')
        return struct.pack('<h',len(byte_string)) + byte_string

@dataclass(frozen=True)
class Int:
    ''''This is the implementation of the little endian 4 byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def to_bytes(self) -> bytes:
        return self.value.to_bytes(4,'little')

@dataclass(frozen=True)
class Varint:
    ''''This is the implementation of the single byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def to_bytes(self) -> bytes:
        return self.value.to_bytes(1, 'little')

@dataclass(frozen=True)
class Varstring:
    '''This is the implementation of a varstring with a one byte integer 
    length prepended.
    '''
    value: str

    def to_bytes(self) -> bytes:
        return Varint(len(self.value)).to_bytes() + bytes(self.value, 'utf8')

@dataclass(frozen=True)
class Double:
    ''''This is the implementation of a double precision float in the 
    eventio bytestream.
    '''
    value: np.float64

    def to_bytes(self) -> bytes:
        return struct.pack('<d', self.value)

@dataclass(frozen=True)
class Short:
    ''''This is the implementation of a double precision float in the 
    eventio bytestream.
    '''
    value: int

    def to_bytes(self) -> bytes:
        return struct.pack('<h', self.value)

