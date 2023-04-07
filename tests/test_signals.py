import CHASM as ch
import numpy as np
import pytest

def test_GH_shower(sample_GH_params):
    '''This test creates a GH shower object and tests the profile method to
    make sure it produces valid particle count values (no nan or negative).
    '''
    shower = ch.MakeGHShower(*sample_GH_params.values())
    Xs = np.linspace(0,1000,100)
    nch = shower.profile(Xs)
    assert np.size(nch[np.isnan(nch)]) == 0
    assert np.size(nch[nch < 0.]) == 0

def test_user_shower(sample_usershower_params):
    '''This test creates a user shower object and tests the profile method to
    make sure it produces valid particle count values (no nan or negative).
    '''
    shower = ch.MakeUserShower(*sample_usershower_params.values())
    Xs = np.linspace(0,1000,100)
    nch = shower.profile(Xs)
    assert np.size(nch[np.isnan(nch)]) == 0
    assert np.size(nch[nch < 0.]) == 0

# def test_downward_axis_fpatm(sample_downward_axis_params):
#     '''This test creates a downward axis and tests its key methods
#     '''

def signal_from_ingredients(shower:str,axis:str,counters:str,cyield:str,request):
    '''This function creaates a basic signal from the fixture names'''
    s = request.getfixturevalue(shower)
    a = request.getfixturevalue(axis)
    c = request.getfixturevalue(counters)
    y = request.getfixturevalue(cyield)
    return ch.Signal(s,a,c,y)

def meshsignal_from_ingredients(shower:str,axis:str,counters:str,cyield:str,request):
    '''This function creaates a basic signal from the fixture names'''
    linear_shower = request.getfixturevalue(shower)
    linear_axis = request.getfixturevalue(axis)
    a = ch.MeshAxis((-6.,0.),linear_axis,linear_shower)
    s = ch.MeshShower(a)
    c = request.getfixturevalue(counters)
    y = request.getfixturevalue(cyield)
    return ch.Signal(s,a,c,y)

def signal_checks(signal: ch.Signal):
    '''This function performs the checks on a signal object'''
    ng = signal.calculate_ng()[0]
    assert np.size(ng[np.isnan(ng)]) == 0
    assert np.size(ng[ng < 0.]) == 0

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_downward_fp_axis","sample_downward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array","sample_flat_ground_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_downward_shower_signal(shower,axis,counters,cyield,request):
    '''This test creates a signal for a downward shower with flat planar atm and
    the various counters.
    '''
    signal_checks(signal_from_ingredients(shower,axis,counters,cyield,request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_downward_fp_axis","sample_downward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array","sample_flat_ground_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_downward_mesh_shower_signal(shower,axis,counters,cyield,request):
    '''This test creates a signal for a downward shower with flat planar atm and
    the various counters.
    '''
    signal_checks(meshsignal_from_ingredients(shower,axis,counters,cyield,request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_upward_fp_axis","sample_upward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array","sample_flat_orbital_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_upward_shower_signal(shower,axis,counters,cyield,request):
    '''This test creates a signal for a downward shower with flat planar atm and
    the various counters.
    '''
    signal_checks(signal_from_ingredients(shower,axis,counters,cyield,request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_upward_fp_axis","sample_upward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array","sample_flat_orbital_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_upward_mesh_shower_signal(shower,axis,counters,cyield,request):
    '''This test creates a signal for a downward shower with flat planar atm and
    the various counters.
    '''
    signal_checks(meshsignal_from_ingredients(shower,axis,counters,cyield,request))

def timing_from_ingredients(axis: str, counters: str, request):
    '''This function creates a timing object from the fixture names.'''
    a = request.getfixturevalue(axis)
    c = request.getfixturevalue(counters)
    return a.get_timing(c)

def mesh_timing_from_ingredients(shower: str, axis: str, counters: str, request):
    '''This function creates a mesh timing object from the fixture names.'''
    linear_shower = request.getfixturevalue(shower)
    linear_axis = request.getfixturevalue(axis)
    a = ch.MeshAxis((-6.,0.),linear_axis,linear_shower)
    s = ch.MeshShower(a)
    c = request.getfixturevalue(counters)
    return a.get_timing(c)

def timing_checks(timing: ch.Timing):
    '''This function performs the check on a Timing object.'''
    t = timing.counter_time()
    assert np.size(t[np.isnan(t)]) == 0

@pytest.mark.parametrize("axis",["sample_downward_fp_axis","sample_downward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array","sample_flat_ground_array"])
def test_downward_timing(axis, counters, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    timing_checks(timing_from_ingredients(axis, counters, request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_downward_fp_axis","sample_downward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array","sample_flat_ground_array"])
def test_downward_mesh_timing(shower, axis, counters, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    timing_checks(mesh_timing_from_ingredients(shower, axis, counters, request))

@pytest.mark.parametrize("axis",["sample_upward_fp_axis","sample_upward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array","sample_flat_orbital_array"])
def test_upward_timing(axis, counters, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    timing_checks(timing_from_ingredients(axis, counters, request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_upward_fp_axis","sample_upward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array","sample_flat_orbital_array"])
def test_upward_mesh_timing(shower, axis, counters, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    timing_checks(mesh_timing_from_ingredients(shower, axis, counters, request))

def attenuation_from_ingredients(axis: str, counters: str, cyield:str, request):
    '''This function creates an attenuation object from the fixture names.'''
    a = request.getfixturevalue(axis)
    c = request.getfixturevalue(counters)
    y = request.getfixturevalue(cyield)
    return a.get_attenuation(c, y)

def mesh_attenuation_from_ingredients(shower: str, axis: str, counters: str, cyield:str, request):
    '''This function creates a mesh attenuation object from the fixture names.'''
    linear_shower = request.getfixturevalue(shower)
    linear_axis = request.getfixturevalue(axis)
    a = ch.MeshAxis((-6.,0.),linear_axis,linear_shower)
    s = ch.MeshShower(a)
    c = request.getfixturevalue(counters)
    y = request.getfixturevalue(cyield)
    return a.get_attenuation(c, y)

def attenuation_checks(attenuation: ch.Attenuation):
    '''This function performs the check on an attenuation object.'''
    f = attenuation.fraction_passed()[0]
    assert np.size(f[np.isnan(f)]) == 0

@pytest.mark.parametrize("axis",["sample_downward_fp_axis","sample_downward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array","sample_flat_ground_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_downward_attenuation(axis, counters, cyield, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    attenuation_checks(attenuation_from_ingredients(axis, counters, cyield, request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_downward_fp_axis","sample_downward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array","sample_flat_ground_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_downward_mesh_attenuation(shower, axis, counters, cyield, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    attenuation_checks(mesh_attenuation_from_ingredients(shower, axis, counters, cyield, request))

@pytest.mark.parametrize("axis",["sample_upward_fp_axis","sample_upward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array","sample_flat_orbital_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_upward_attenuation(axis, counters, cyield, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    attenuation_checks(attenuation_from_ingredients(axis, counters, cyield, request))

@pytest.mark.parametrize("shower",["sample_GH_shower","sample_user_shower"])
@pytest.mark.parametrize("axis",["sample_upward_fp_axis","sample_upward_curved_axis"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array","sample_flat_orbital_array"])
@pytest.mark.parametrize("cyield",["sample_yield"])
def test_upward_mesh_attenuation(shower, axis, counters, cyield, request):
    '''This test creates the various possibilities for an upward timing object
    and performs the checks in timing_checks().'''
    attenuation_checks(mesh_attenuation_from_ingredients(shower, axis, counters, cyield, request))
