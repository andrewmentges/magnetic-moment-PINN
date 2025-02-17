***
# Magnetic Dipole Moment and Multipole Determination From Near-Field Data via Physics Informed Neural Networks
***

### By Andrew Mentges
### amentges@captechu.edu

Physics informred neural network for performing magnetic dipole moment and magnetic multipole moment determination

****
# Simulator
***
The `smtf_utility.py` script contians several helper functions. One of the main functions is the B-field simulator, `bf_simulator` class. This class can be 
used to create a sample test data set by supplying magnetic moments and various locations within the test area and at various strengths. The output data will
be in the standard format of position value pairs of observed field measurements.

## Constructor
`bf_simulator(distances, delta, scale=1)`
* `distances` - numpy list of distances in meters along the X-Axis where test measurements would be made. Typical distances would be `[1, 1.33, 1.66, 1.99]`
* `delta` - The angle of rotation in degrees. For example, a test where the instrument is rotated 30 degrees at every measurement point would have a delta  of 30.
* `scale` - This is used to scale the output B-field strength into the units of interest. Typical test data is scaled to nano-Tesla, or 1e9, for easier convergence.

## Methods
`add_dipole(mlocation_x, mlocation_y, mlocation_z, mval_x, mval_y, mval_z)`

Method that will generate B-Field data from the supplied dipole moment information and then add those generated fields to the internally stored B-Field data
* `mlocation_x` - The X-axis position of the dipole moment to be added. Units are in meters. Value should be inside the area described by the distances.
* `mlocation_y` - The Y-axis position of the dipole moment to be added. Units are in meters. Value should be inside the area described by the distances.
* `mlocation_z` - The Z-axis position of the dipole moment to be added. Units are in meters. Value should be inside the area described by the distances.
* `mval_x` - The X-axis value of the dipole moment to be added. Units are in $Amp\cdot meters^{2}$
* `mval_y` - The Y-axis value of the dipole moment to be added. Units are in $Amp\cdot meters^{2}$
* `mval_z` - The Z-axis value of the dipole moment to be added. Units are in $Amp\cdot meters^{2}$

`plot(axis, plevels)`

Method for plotting the internal B-Field for a specific axis
* `axis` - Axis to be plotted. Values can be "X", "Y", "Z". For example, `axis="X"`
* `plevels` - Number of contour levels for the plot that will be generated

`get_data`

Method for getting ordered training data. Returns a tuple of numpy arrays such that Position, BField = `model.get_data()`

Where Position about have the format of [[X1, Y1, Z1], [X2, Y2, Z2],...] and BField would have the format of [[Bx1, By1, Bz1], [Bx2, By2, Bz2],...]

Position will have the units of meters and Bfield values would have the units of 1 / `scale`.
