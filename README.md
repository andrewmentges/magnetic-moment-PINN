***
# Magnetic Dipole Moment and Multipole Determination From Near-Field Data via Physics Informed Neural Networks
***

### By Andrew Mentges
### amentges@captechu.edu

Physics informed neural network for performing magnetic dipole moment and magnetic multipole moment determination.
The code here was developed with the following libraries:
* Tensorflow version 2.15.0
* Python 3.11
* Jupyter notebooks
* matplotlib
* numpy

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


****
# Multipole Model
***
This model is a physics informed neural network that can be used to determine the magnetic multipole system of an a device under test. The model
allows the user to tweaker various hyper-parameters in order to support convergence to a solution. Solved data is presented as a list of 
position value pairs of magnetic moments. The posistions are in units of meters and the values are in units of $Amp\cdot meters^{2}$.

## Constructor
`MultiPoleModel(moments=1, lrate=.01, optimizer='adam', loss='mse', scale=1, early_stop=False, target_stop=1)`
* `moments` - The number of magnetic moment layers the model will use to solve for. The closer the number of moments to the actual number of moments
  used to generate the magnetic field test data, the more accurate the model will be.
* `lrate` - This is the learning rate of the model. This is typically on the order of 0.01. But this value can and should change depending on the optimizer chosen.
* `optimizer` - This is the optimizer used in the training of the model. The supported optimizers are: `sgd`, `rmsprop`, `adam`, `nadam`, `adadelta`, `adagrad`.
* `loss` - This is the loss function used in the training of the model. The supported loss functions are: `mae`, `mse`, `huber`
