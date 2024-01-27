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

def showersignal_checks(sig: ch.ShowerSignal):
    arrs2check =   [sig.source_points,
                    sig.wavelengths,
                    sig.photons,
                    sig.times,
                    sig.charged_particles,
                    sig.depths,
                    sig.total_photons,
                    sig.cos_theta]
    for arr in arrs2check:
        assert np.isnan(arr).any() == False

def sim_from_inputs(shower_input:str,axis_input:str,counters_input:str,cyield_input:str,request):
    sim = ch.ShowerSimulation()
    sim.add(request.getfixturevalue(shower_input))
    sim.add(request.getfixturevalue(axis_input))
    sim.add(request.getfixturevalue(counters_input))
    sim.add(request.getfixturevalue(cyield_input))
    return sim

@pytest.mark.parametrize("shower",["sample_GH_shower_input","sample_user_shower_input"])
@pytest.mark.parametrize("axis",["sample_downward_fp_axis_input","sample_downward_curved_axis_input"])
@pytest.mark.parametrize("counters",["sample_spherical_ground_array_input","sample_flat_ground_array_input"])
@pytest.mark.parametrize("cyield",["sample_yield_input"])
@pytest.mark.parametrize("mesh",[True, False])
@pytest.mark.parametrize("att",[True, False])
def test_downward_run(shower,axis,counters,cyield,mesh,att,request):
    '''This test creates a sim for a downward shower, runs it, and checks the output.
    '''
    sim = sim_from_inputs(shower,axis,counters,cyield,request)
    sig = sim.run(mesh=mesh,att=att)
    showersignal_checks(sig)

@pytest.mark.parametrize("shower",["sample_GH_shower_input","sample_user_shower_input"])
@pytest.mark.parametrize("axis",["sample_upward_fp_axis_input","sample_upward_curved_axis_input"])
@pytest.mark.parametrize("counters",["sample_spherical_orbital_array_input","sample_flat_orbital_array_input"])
@pytest.mark.parametrize("cyield",["sample_yield_input"])
@pytest.mark.parametrize("mesh",[True, False])
@pytest.mark.parametrize("att",[True, False])
def test_upward_run(shower,axis,counters,cyield,mesh,att,request):
    '''This test creates a sim for a downward shower, runs it, and checks the output.
    '''
    sim = sim_from_inputs(shower,axis,counters,cyield,request)
    sig = sim.run(mesh=mesh,att=att)
    showersignal_checks(sig)
