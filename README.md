![CHASM_logo](https://user-images.githubusercontent.com/64815713/217383767-9bdf9ff9-88ec-43f5-b670-c92de67aa085.png)

# CHASM (CHerenkov Air Shower Model)

CHASM is a python package which leverages the universality of charged particles in an extensive air shower to produce a deterministic prediction of the Cherenkov light signal for a given shower profile and geometry. At samples throughout the domain of all shower development stages and altitudes, the angular and yield distributions of Cherenkov light have been calculated at an array of distances from a shower axis. Chasm accesses and interpolates between these distributions at runtime to produce the aggregate signal from the whole shower at user defined telescope locations.

# Installation

To install from pip:

`pip install CHASM-NuSpacesim`￼

To install from source:

1. `git clone https://github.com/ikepc1/CHASM_NuSpacesim`
2. `cd CHASM_NuSpacesim`
3. `python3 -m pip install -e .`

# Usage

CHASM has implementations of both upward and downward going shower axes in either flat planar or curved atmospheres. A shower profile can be defined either by Gaisser-Hillas parameters, or as an array of particle counts and at a corresponding array of depths. Cherenkov yield distributions can be sampled along the shower axis, or in a mesh of points surrounding the axis, at which charged particles are distributed according to the NKG lateral distribution. The first step is to create a simulation.

```
from CHASM.simulation import *

sim = ShowerSimulation()
```

Then add an axis. The origin of the coordinate system is where the axis meets the ground. It is defined by a polar angle, azimuthal angle, and keyword arguments for the ground level and whether to account for atmospheric curvature.
```
sim.add(DownwardAxis(theta, phi, ground_level = 0., curved = False))
```
or
```
sim.add(UpwardAxis(theta, phi, ground_level = 0., curved = False))
```

Then add a shower.

```
sim.add(GHShower(X_max, N_max, X0, Lambda))
```
or
```
sim.add(UserShower(X, N))
```

Now we add photon counters. Both spherical CORSIKA IACT style counters and flat counting apertures are available.

```
sim.add(SphericalCounters(counter_vectors, counter_radius))
```
or
```
sim.add(FlatCounters(counter_vectors, counter_radius))
```

Finally, we define the Cherenkov wavelength interval of interest.

```
sim.add(Yield(min_l, max_l, N_bins = 1))
```

We can now run the simulation, and generate signals.

```
sim.run(mesh = True)
sim.get_signal_sum()
sim.get_attenuated_signal_sum()

times = sim.get_signal_times()
photons = sim.get_signal_times(att=False)
```
