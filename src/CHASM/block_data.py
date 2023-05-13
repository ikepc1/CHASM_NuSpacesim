from dataclasses import dataclass, field
from datetime import datetime
import struct
import numpy as np

from .eventio_types import EventioType, Float, Double, String, Int, Varint, Varstring, Short, PhotonsData, ThreeByte
from .config import AxisConfig
from .simulation import ShowerSimulation, ShowerSignal
from .axis import Axis, Counters
from .shower import Shower
from .generate_Cherenkov import MakeYield

def date_to_float() -> Float:
    '''This function takes the current date and returns it as a float in YYMMDD
    format for the run header.
    '''
    return Float(float(datetime.now().strftime("%y%m%d")))

def array_to_eventio_list(arr: np.ndarray, type: EventioType) -> list[EventioType]:
    '''This function takes a 2d numpy array and converts the values into
    a list of corresponding eventiotypes in the correct order for use in the
    bytestream.
    '''
    return [type(val) for val in arr.flatten().tolist()]

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
    nwords: Int = Int(273)
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
    nstrings: Int = Int(1)
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
    return array_to_eventio_list(atm_array, Double)

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
    empty_word: ThreeByte
    tel_array: np.ndarray
    # tel_x: list[Float]
    # tel_y: list[Float]
    # tel_z: list[Float]
    # tel_r: list[Float]

def make_tel_def(sig: ShowerSignal) -> TelescopeDefinitionData:
    '''This function returns an instantiated Telescope definition
     data block container.
    '''
    vec_cm = np.array(sig.counters.vectors.T.flatten()*100, dtype=np.float32)
    r_cm = np.array(sig.counters.input_radius*100, dtype=np.float32)
    ta = np.append(vec_cm,r_cm)
    # x = [Float(val) for val in vectors_cm[:,0].tolist()]
    # y = [Float(val) for val in vectors_cm[:,1].tolist()]
    # z = [Float(val) for val in vectors_cm[:,2].tolist()]
    # r = [Float(val) for val in np.full(sig.counters.N_counters,sig.counters.input_radius*100.).tolist()]
    return TelescopeDefinitionData(Varint(sig.counters.N_counters),ThreeByte(0),ta)

@dataclass
class EventHeaderData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    eventio event header block.
    '''
    nwords: Int = Int(273)
    evth: Float = Float(struct.unpack('f',b'EVTH')[0])
    evno: Float = Float(1.)
    p_id: Float = Float(1.)
    total_energy: Float = Float(1.e8)
    starting_altitude: Float = Float(120.e6)
    first_target_id: Float = Float(1.)
    first_interaction_height: Float = Float(120.e6)
    momentum_x: Float = Float(1.)
    momentum_y: Float = Float(1.)
    momentum_minus_z: Float = Float(1.)
    zenith: Float = Float(0.)
    azimuth: Float = Float(0.)
    n_random_sequences: Float = Float(1.)
    random_seeds: list[Float] = field(default_factory= lambda: [Float(1.)] * 30)
    run_number: Float = Float(1.)
    date: Float = field(default_factory=date_to_float) #Date as float YYMMDD.
    version: Float = Float(7.741) #mock CORSIKA version
    n_observation_levels: Float = Float(1)
    obs_levels: list[Float] = field(default_factory= lambda: [Float(0.)] * 10) #observation levels
    e_slope: Float = Float(-1.) #-1.
    e_low: Float = Float(1.e8) #Lower limit of energy range (primary)
    e_high: Float = Float(1.e8) #Upper limit of energy range (primary)
    energy_cutoff_hadrons: Float = Float(0.)
    energy_cutoff_muons: Float = Float(0.)
    energy_cutoff_electrons: Float = Float(0.)
    energy_cutoff_photons: Float = Float(0.)
    nflain: Float = Float(0.)
    nfdif: Float = Float(0.)
    nfdif: Float = Float(0.)
    nflpi0: Float = Float(0.)
    nflpif: Float = Float(0.)
    nflche: Float = Float(0.)
    nfragm: Float = Float(0.)
    earth_magnetic_field_x: Float = Float(0.)
    earth_magnetic_field_z: Float = Float(0.)
    egs4_flag: Float = Float(0.)
    nkg_flag: Float = Float(0.)
    low_energy_hadron_model: Float = Float(0.)
    high_energy_hadron_model: Float = Float(0.)
    cerenkov_flag: Float = Float(0.)
    neutrino_flag: Float = Float(0.)
    curved_flag: Float = Float(0.)
    theta_min: Float = Float(0.)
    theta_max: Float = Float(0.)
    phi_min: Float = Float(0.)
    phi_max: Float = Float(0.)
    cherenkov_bunch_size: Float = Float(1.)
    n_cherenkov_detectors_x: Float = Float(1.)
    n_cherenkov_detectors_y: Float = Float(1.)
    cherenkov_detector_grid_spacing_x: Float = Float(1.)
    cherenkov_detector_grid_spacing_y: Float = Float(1.)
    cherenkov_detector_length_x: Float = Float(1.)
    cherenkov_detector_length_y: Float = Float(1.)
    cherenkov_output_flag: Float = Float(1.)
    angle_array_x_magnetic_north: Float = Float(1.)
    additional_muon_information_flag: Float = Float(1.)
    egs4_multpliple_scattering_step_length_factor: Float = Float(1.)
    cherenkov_wavelength_min: Float = Float(200.)
    cherenkov_wavelength_max: Float = Float(900.)
    n_reuse: Float = Float(0.)
    reuse_x: list[Float] = field(default_factory= lambda: [Float(0.)] * 20)
    reuse_y: list[Float] = field(default_factory= lambda: [Float(0.)] * 20)
    sybill_interaction_flag: Float = Float(0.)
    sybill_cross_section_flag: Float = Float(0.)
    qgsjet_interaction_flag: Float = Float(0.)
    qgsjet_cross_section_flag: Float = Float(0.)
    dpmjet_interaction_flag: Float = Float(0.)
    dpmjet_cross_section_flag: Float = Float(0.)
    venus_nexus_epos_cross_section_flag: Float = Float(0.)
    muon_multiple_scattering_flag: Float = Float(0.)
    nkg_radial_distribution_range: Float = Float(0.)
    energy_fraction_if_thinning_level_hadronic: Float = Float(0.)
    energy_fraction_if_thinning_level_em: Float = Float(0.)
    actual_weight_limit_thinning_hadronic: Float = Float(0.)
    actual_weight_limit_thinning_em: Float = Float(0.)
    max_radius_radial_thinning_cutting: Float = Float(0.)
    viewcone_inner_angle: Float = Float(0.)
    viewcone_outer_angle: Float = Float(0.)
    transition_energy_low_high_energy_model: Float = Float(0.)
    later_versions_placeholders: list[Float] = field(default_factory= lambda: [Float(0.)] * 119)

def make_event_header(sig: ShowerSignal) -> EventHeaderData:
    '''This function makes an event header container for a CHASM
    simulation.
    '''
    return EventHeaderData(
        zenith = Float(sig.axis.zenith),
        azimuth = Float(sig.axis.azimuth),
        obs_levels = [Float(sig.counters.vectors[:,2].min())] * 10
    )

@dataclass
class ArrayOffsetsData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    array offsets block.
    '''
    n_offsets: Int = Int(1)
    t_offset: Float = Float(0)
    x_offset: Float = Float(0)
    y_offset: Float = Float(0)

def make_array_offsets(sig: ShowerSignal) -> ArrayOffsetsData:
    '''This function extracts the time offset from a CHASM sim for 
    use in the ArrayOffsets datablock.
    '''
    return ArrayOffsetsData()

@dataclass
class LongitudinalData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    longitudinal block.
    '''
    event_id: Int = Int(1)
    type: Int = Int(1)
    np: Short = Short(2) #CHASM only deals in charged particles and cherenkov photons (2 dists)
    nthick: Short = Short(1000)
    thickstep: Float = Float(1)
    nch: list[Float] = field(default_factory= lambda: [Float(0.)] * 1000)
    ng: list[Float] = field(default_factory= lambda: [Float(0.)] * 1000)

def make_longitudinal(sig: ShowerSignal) -> LongitudinalData:
    '''This function extracts the shower profile and total Cherenkov to include in the
    mock longitudinal data block.
    '''
    X = sig.depths
    N_thick = int(np.floor(X.max()))
    Xs = np.arange(N_thick)

    #depending on upward vs downward axis, the depths will increasing or decreasing
    #respectively. Numpy interp needs strictly increasing x values.
    if np.all(np.diff(X) > 0.):
        nch_1g = np.interp(Xs,X,sig.charged_particles).tolist()
        ng_1g = np.interp(Xs,X,sig.total_photons).tolist()
    else:
        nch_1g = np.interp(Xs,X[::-1],sig.charged_particles[::-1]).tolist()
        ng_1g = np.interp(Xs,X[::-1],sig.total_photons[::-1]).tolist()
    n_g = [Float(val) for val in ng_1g]
    N_ch = [Float(val) for val in nch_1g]
    return LongitudinalData(nthick=Short(N_thick), nch=N_ch, ng=n_g)

@dataclass
class TelescopeData:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    telescope events block.
    '''
    tel_data: list[PhotonsData]

def make_telescope_data(sig: ShowerSignal) -> TelescopeData:
    '''This function constructs a TelescopeData object from a shower signal
    object.
    '''
    tel_data = []
    for i in range(sig.counters.N_counters):
        bunches = sig.get_bunches(i)
        tel_data.append(PhotonsData(
            id = Int(i),
            length = Int(bunches.size*4 + 12), #4 bytes per float 12, in the header
            tel_no = Short(i),
            n_photons = Float(sig.photons[i].sum()),
            n_bunches = Int(bunches.shape[0]),
            bunches = bunches
        ))
    return TelescopeData(tel_data)

@dataclass
class EventEnd:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    event end block.
    '''
    nwords: Int = Int(273)
    evte: Float = Float(struct.unpack('f',b'EVTE')[0])
    evt_no: Float = Float(1.)
    n_photons: Float = Float(0.)
    n_electrons: Float = Float(0.)
    empty_words: list[Float] = field(default_factory= lambda: [Float(0.)] * 269)
    
def make_event_end(sig: ShowerSignal) -> EventEnd:
    '''This finction makes an event end data block.
    '''
    return EventEnd(n_photons = Float(sig.total_photons.sum()),
                    n_electrons = Float(sig.charged_particles.sum()))

@dataclass
class RunEnd:
    '''This class contains all the parameters needed to construct a mock CORSIKA
    run end block.
    '''
    nwords: Int = Int(3)
    rune: Float = Float(struct.unpack('f',b'RUNE')[0])
    run_no: Float = Float(1.)
    n_events: Float = Float(1.)
