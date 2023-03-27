import struct

def float_to_bytes(input_float: float) -> bytes:
    '''This function returns the little endian 4 byte floats found in the eventio
    bytestream.'''
    return struct.pack('<f', input_float)

def string_to_bytes(input_string: str) -> bytes:
    '''This function returns an eventio style string byte buffer with the 
    length prepended as a 2 byte unsigned short.
    '''
    byte_string = bytes(input_string, 'utf8')
    return struct.pack('<h',len(byte_string)) + byte_string