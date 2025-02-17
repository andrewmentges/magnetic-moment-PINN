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


Code for the physics informed neural network models below can be found in `pinn_magnetic_experimental.py`. Samples of how you use the models can be seen 
in the jupyter notebooks found within the repository. This application was originally developed for spacecraft testing but can be used for any application
that needs to determine the magnetic moment makeup of any article that can be tested.

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

Method that will generate B-field data from the supplied dipole moment information and then add those generated fields to the internally stored B-field data.
* `mlocation_x` - The X-axis position of the dipole moment to be added. Units are in meters. Value should be inside the area described by the distances.
* `mlocation_y` - The Y-axis position of the dipole moment to be added. Units are in meters. Value should be inside the area described by the distances.
* `mlocation_z` - The Z-axis position of the dipole moment to be added. Units are in meters. Value should be inside the area described by the distances.
* `mval_x` - The X-axis value of the dipole moment to be added. Units are in $Amp\cdot meters^{2}$
* `mval_y` - The Y-axis value of the dipole moment to be added. Units are in $Amp\cdot meters^{2}$
* `mval_z` - The Z-axis value of the dipole moment to be added. Units are in $Amp\cdot meters^{2}$

`plot(axis, plevels)`

Method for plotting the internal B-field for a specific axis.
* `axis` - Axis to be plotted. Values can be "X", "Y", "Z". For example, `axis="X"`
* `plevels` - Number of contour levels for the plot that will be generated

`get_data`

Method for getting ordered training data. Returns a tuple of numpy arrays such that Position, B-field = `model.get_data()`

Where Position about have the format of [[ $X_{1}$, $Y_{1}$, $Z_{1}$], [ $X_{2}$, $Y_{2}$, $Z_{2}$],...] and B-field would have the format of [[ $B_{X_1}$, $B_{Y_1}$, $B_{Z_1}$], 
[ $B_{X_2}$, $B_{Y_2}$, $B_{Z_2}$],...] Position will have the units of meters and Bfield values would have the units of $\frac{1} {scale}$.


****
# Multipole Model
***
This model is a physics informed neural network that can be used to determine the magnetic multipole moment system of an a device under test. The model
allows the user to tweak various hyper-parameters in order to support convergence to a solution. Solved data is presented as a list of 
position value pairs of magnetic moments. The positions are in units of meters and the values are in units of $Amp\cdot meters^{2}$.

## Constructor
`MultiPoleModel(moments=1, lrate=.01, optimizer='adam', loss='mse', scale=1, early_stop=False, target_stop=1)`
* `moments` - The number of magnetic moment layers the model will use to solve for. The closer the number of moments to the actual number of moments
  used to generate the magnetic field test data, the more accurate the model will be.
* `lrate` - This is the learning rate of the model. This is typically on the order of 0.01. But this value can and should change depending on the optimizer chosen.
* `optimizer` - This is the optimizer used in the training of the model. The supported optimizers are: `sgd`, `rmsprop`, `adam`, `nadam`, `adadelta`, `adagrad`.
* `loss` - This is the loss function used in the training of the model. The supported loss functions are: `mae`, `mse`, `huber`.
* `scale` - Determines the scale used for the training. A scale of 1, the input training data would have units of Tesla. For 1e9, the training data would be in the
  units of nano-Tesla, which is typical for this type of testing. Matching the scale of the training data to the model scale is necessary for getting accurate results.
* `early_stop` -  If a value of `True` is passed, the model will stop training the after the loss of the current epcoh of training is higher than the previous epoch.
* `target_stop` - This can be used to stop the training of the model when a specific value of loss has been reach. For `mae` and `mse` this can be related to a specific
  range magnetic field error in matching the average epoch loss. A standard value of 1 will give an extremely accurate solution if reached for any of the supported losses.

## Methods
`fit(positions, values, epochs)`

Method used to train the model against collected data.
* `positions` - A list `[]` of positions where the magnetic field is observed. It has units of meters and the format of [[ $X_{1}$, $Y_{1}$, $Z_{1}$], [ $X_{2}$, $Y_{2}$, $Z_{2}$],...].
* `values` - A list `[]` of B-field values for the observed positions that should have the format of [[ $B_{X_1}$, $B_{Y_1}$, $B_{Z_1}$], [ $B_{X_2}$, $B_{Y_2}$, $B_{Z_2}$],...].
* `epochs` - The number of times that the positions and values should be iterated over to train the model. The number could vary from 100 to 2000.

`moment()`

Method used print out the position value pairs of the magnetic moments that have been determined by the model after training. Positions have units of meters and values are in $Amp\cdot meters^{2}$.


****
# MultiDipole Model
***
This model is a physics informed neural network that can be used to determine the magnetic dipole moment of an a device under test. The model
allows the user to tweak various hyper-parameters in order to support convergence to a solution. Solved data is presented as a list of 
x, y, z field strengths, [ $B_{X}$, $B_{Y}$, $B_{Z}$ ] centered about the origin of the test area. Field strength is in units of $Amp\cdot meters^{2}$.

## Constructor
`MultiDipoleModel(poles=1, lrate=1000, optimizer='adam', loss='mse', scale=1, early_stop=False, target_stop=1)`
* `poles` - The number of magnetic dipole moment layers the model will use to solve for. For simple fields a value of 1 can be used. If the multipole system is
 sufficiently complex, more dipole moment layers can be stacked on each other to develop a more accurate solution.
* `lrate` - This is the learning rate of the model. This is typically on the order of 0.01. But this value can and should change depending on the optimizer chosen.
* `optimizer` - This is the optimizer used in the training of the model. The supported optimizers are: `sgd`, `rmsprop`, `adam`, `nadam`, `adadelta`, `adagrad`.
* `loss` - This is the loss function used in the training of the model. The supported loss functions are: `mae`, `mse`, `huber`.
* `scale` - Determines the scale used for the training. A scale of 1, the input training data would have units of Tesla. For 1e9, the training data would be in the
  units of nano-Tesla, which is typical for this type of testing. Matching the scale of the training data to the model scale is necessary for getting accurate results.
* `early_stop` -  If a value of `True` is passed, the model will stop training the after the loss of the current epcoh of training is higher than the previous epoch.
* `target_stop` - This can be used to stop the training of the model when a specific value of loss has been reach. For `mae` and `mse` this can be related to a specific
  range magnetic field error in matching the average epoch loss. A standard value of 1 will give an extremely accurate solution if reached for any of the supported losses.

## Methods
`fit(positions, values, epochs)`

Method used to train the model against collected data.
* `positions` - A list `[]` of positions where the magnetic field is observed. It has units of meters and the format of [[ $X_{1}$, $Y_{1}$, $Z_{1}$], [ $X_{2}$, $Y_{2}$, $Z_{2}$],...].
* `values` - A list `[]` of B-field values for the observed positions that should have the format of [[ $B_{X_1}$, $B_{Y_1}$, $B_{Z_1}$], [ $B_{X_2}$, $B_{Y_2}$, $B_{Z_2}$],...].
* `epochs` - The number of times that the positions and values should be iterated over to train the model. The number could vary from 100 to 2000.

`dipole()`

Method used print out the value of the magnetic dipole moment that was determined by the model after training. The output is in the format of
[ $B_{X}$, $B_{Y}$, $B_{Z}$ ] and has units of $Amp\cdot meters^{2}$.
