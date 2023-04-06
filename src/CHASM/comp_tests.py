import eventio
import struct
from CHASM.write_eventio import *

testfile = "/home/isaac/corsika_data/correct_thinning/iact_DAT000001.dat"

ei = eventio.IACTFile(testfile)
ei._next_header_pos = 0
with open(testfile,'rb') as open_file:
    bb = open_file.read()
