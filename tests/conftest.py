import CHASM as ch

import numpy as np
from scipy.stats import norm
import pytest

@pytest.fixture
def sample_GH_params():
    '''Sample GH parameters to create a test GH MakeShower object'''
    return {
    'X_max':666.,
    'N_max':1.e6,
    'X0':0.,
    'Lambda':70.
    }

@pytest.fixture
def sample_GH_shower(sample_GH_params):
    '''Return an instantiated GH shower object.'''
    return ch.MakeGHShower(*sample_GH_params.values())

@pytest.fixture
def sample_usershower_params():
    '''Sample GH parameters to create a test GH MakeShower object'''
    X = np.linspace(0,1000,100)
    N_ch = 1.e9 * norm.pdf(X, loc=500, scale=50)
    return {
    'X':X,
    'N_ch':N_ch
    }

@pytest.fixture
def sample_user_shower(sample_usershower_params):
    '''Return an instantiated User shower object.'''
    return ch.MakeUserShower(*sample_usershower_params.values())

@pytest.fixture
def sample_downward_axis_params():
    '''Sample parameters for downward axis with flat-planar atm'''
    return {
    'zenith':np.radians(30.),
    'azimuth':np.radians(45.),
    'ground_level':0.
    }


@pytest.fixture
def sample_downward_fp_axis(sample_downward_axis_params):
    '''Return instantiated downward axis with fp atm axis.'''
    return ch.DownwardAxis(*sample_downward_axis_params.values()).create()

@pytest.fixture
def sample_downward_curved_axis(sample_downward_axis_params):
    '''Return instantiated downward axis with curved atm axis.'''
    
    return ch.DownwardAxis(*sample_downward_axis_params.values(),curved=True).create()

@pytest.fixture
def sample_upward_axis_params():
    '''Sample parameters for downward axis with flat-planar atm'''
    return {
    'zenith':np.radians(85.),
    'azimuth':np.radians(0.),
    'ground_level':0.
    }

@pytest.fixture
def sample_upward_fp_axis(sample_upward_axis_params):
    '''Returns instantiated upward axis with fp atm.'''
    return ch.UpwardAxis(*sample_upward_axis_params.values()).create()

@pytest.fixture
def sample_upward_curved_axis(sample_upward_axis_params):
    '''Returns instantiated upward axis with curved atm.'''
    return ch.UpwardAxis(*sample_upward_axis_params.values(),curved=True).create()

@pytest.fixture
def sample_ground_array_geometry():
    '''sample vectors to ground array locations.'''
    x = np.linspace(-1.e4,1.e4,10)
    xx, yy = np.meshgrid(x,x)
    counters = np.empty([xx.size,3])
    counters[:,0] = xx.flatten()
    counters[:,1] = yy.flatten()
    counters[:,2] = np.zeros(xx.size)
    return {
    'vectors':counters,
    'radius':1.
    }

@pytest.fixture
def sample_spherical_ground_array(sample_ground_array_geometry):
    '''Returns instantiated spherical ground array.'''
    return ch.MakeSphericalCounters(*sample_ground_array_geometry.values())

@pytest.fixture
def sample_flat_ground_array(sample_ground_array_geometry):
    '''Returns instantiated flat ground array.'''
    return ch.MakeFlatCounters(*sample_ground_array_geometry.values())

@pytest.fixture
def sample_orbital_array_geometry():
    '''sample vectors to orbital array locations.'''
    r = 2141673.
    theta = np.radians(85.)
    arc = np.radians(2.)
    phi = 0.
    Nxc = 12
    Nyc = 12

    phis = np.linspace(-arc,arc,Nxc)
    thetas = np.linspace(theta-arc,theta+arc,Nyc)
    counters = np.empty([Nxc*Nyc,3])
    theta_tel, phi_tel = np.meshgrid(thetas,phis)

    counters[:,0] = r * np.sin(theta_tel.flatten()) * np.cos(phi_tel.flatten())
    counters[:,1] = r * np.sin(theta_tel.flatten()) * np.sin(phi_tel.flatten())
    counters[:,2] = r * np.cos(theta_tel.flatten())
    return {
    'vectors':counters,
    'radius':1.
    }

@pytest.fixture
def sample_spherical_orbital_array(sample_orbital_array_geometry):
    '''Returns instantiated spherical ground array.'''
    return ch.MakeSphericalCounters(*sample_orbital_array_geometry.values())

@pytest.fixture
def sample_flat_orbital_array(sample_orbital_array_geometry):
    '''Returns instantiated flat ground array.'''
    return ch.MakeFlatCounters(*sample_orbital_array_geometry.values())

@pytest.fixture
def sample_yield_interval():
    '''Sample Cherenkov wavelength interval.'''
    return {
    'min_l':300,
    'max_l':900,
    }

@pytest.fixture
def sample_yield(sample_yield_interval):
    '''Returns list with an instantiated yield object.'''
    return [ch.MakeYield(*sample_yield_interval.values())]
