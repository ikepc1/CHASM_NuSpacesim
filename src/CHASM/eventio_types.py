import struct
from typing import Protocol
from dataclasses import dataclass, fields, field
import numpy as np

class EventioType(Protocol):
    '''This is the base class for an eventio datatype.
    '''

    def tobytes(self) -> bytes:
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

    def tobytes(self) -> bytes:
        return struct.pack('<f', self.value)

@dataclass(frozen=True)
class String:
    '''This is the implementation of an eventio style string with a 2 byte
    short integer representing the length prepended.
    '''
    value: str

    def tobytes(self) -> bytes:
        byte_string = bytes(self.value, 'utf8')
        return struct.pack('<h',len(byte_string)) + byte_string

@dataclass(frozen=True)
class Int:
    ''''This is the implementation of the little endian 4 byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def tobytes(self) -> bytes:
        return self.value.to_bytes(4,'little')

@dataclass(frozen=True)
class Varint:
    ''''This is the implementation of the single byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def tobytes(self) -> bytes:
        return self.value.to_bytes(1, 'little')

@dataclass(frozen=True)
class ThreeByte:
    ''''This is the implementation of the three byte ints found in the 
    eventio bytestream.
    '''
    value: int

    def tobytes(self) -> bytes:
        return self.value.to_bytes(3, 'little')

@dataclass(frozen=True)
class Varstring:
    '''This is the implementation of a varstring with a one byte integer 
    length prepended.
    '''
    value: str

    def tobytes(self) -> bytes:
        return Varint(len(self.value)).tobytes() + bytes(self.value, 'utf8')

@dataclass(frozen=True)
class Double:
    ''''This is the implementation of a double precision float in the 
    eventio bytestream.
    '''
    value: np.float64

    def tobytes(self) -> bytes:
        return struct.pack('<d', self.value)

@dataclass(frozen=True)
class Short:
    ''''This is the implementation of a double precision float in the 
    eventio bytestream.
    '''
    value: int

    def tobytes(self) -> bytes:
        return struct.pack('<h', self.value)
    
@dataclass
class PhotonsData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    photon bunch block.
    '''
    type: Int = Int(1205)
    id: Int = Int(0)
    length: Int = Int(0)
    arr: Short = Short(0)
    tel_no: Short = Short(0)
    n_photons: Float = Float(0.)
    n_bunches: Int = Int(0)
    bunches: np.ndarray = field(default_factory= lambda: np.empty(0))

    def tobytes(self) -> bytes:
        b = bytearray()
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            b.extend(value.tobytes())
        # b.extend(self.type.tobytes())
        # b.extend(self.id.tobytes())
        # b.extend(self.length.tobytes())
        # b.extend(self.arr.tobytes())
        # b.extend(self.tel_no.tobytes())
        # b.extend(self.n_photons.tobytes())
        # b.extend(self.n_bunches.tobytes())
        # b.extend(self.bunches.tobytes())
        return b
