import eventio
import struct
import CHASM as ch
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# testfile = "/home/isaac/corsika_data/60deg/iact_DAT000001.dat"
# testfile = "/home/isaac/corsika_data/unattenuated_30degree_sealevel/iact_s_000001.dat"
testfile = "iact_DAT000001.dat"

ei = eventio.IACTFile(testfile)
ei._next_header_pos = 0
with open(testfile,'rb') as open_file:
    bb = open_file.read()


cors_no_att = ch.EventioWrapper(testfile)

sim = ch.ShowerSimulation()

#Add shower axis
sim.add(ch.DownwardAxis(cors_no_att.theta, cors_no_att.phi, cors_no_att.obs))

#Add shower profile
sim.add(ch.UserShower(cors_no_att.X, cors_no_att.nch)) #profiles could also be created using Gaisser-Hillas parameters

#Add CORSIKA IACT style spherical telescopes
sim.add(ch.SphericalCounters(cors_no_att.counter_vectors, cors_no_att.counter_radius))

#Add wavelength interval for Cherenkov yield calculation
sim.add(ch.Yield(cors_no_att.min_l, cors_no_att.max_l,3))

sig = sim.run(mesh = False)

# b = ch.eventio_bytes(sig)
ch.write_ei_file(sig, 'test.dat')
ei_test = eventio.IACTFile('test.dat')
ei_test._next_header_pos = 0
# count = 0
# for i in range(4168+4,4924,4):
#     count += 1
#     print(struct.unpack('<f',bb[i:i+4]), count)

# count = 0
# bb1 = bytes(b)
# for i in range(21,len(b),4):
#     count += 1
#     print(struct.unpack('<f',bb1[i:i+4]), count)
event_test = next(iter(ei_test))
event = next(iter(ei))

for i in [0,5,10,15,20,25,30]:
    plt.figure()
    plt.scatter(event.photon_bunches[i]['x'],event.photon_bunches[i]['y'],s = .1, label='iact')
    plt.scatter(event_test.photon_bunches[i]['x'],event_test.photon_bunches[i]['y'],s = .1, label='chasm')
    plt.title(f'Counter {i}')
    plt.legend()