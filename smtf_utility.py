# Utilities and helper functions for the Goddard Spaceflight Center Spacecraft Magnetic Test Facility
# Created by Andrew Mentges

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def shuffle_data(positions, values):
    mapIndexPosition = list(zip(positions, values))
    random.shuffle(mapIndexPosition)
    return zip(*mapIndexPosition)

#calculates the magnitude of a vector
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


#rotates x-y coordinates about the z-axis
def rotate(px, py, pz, theta):  
    theta = np.radians(theta)
      
    original_pos = np.array([[px, py, pz]]).T  
    rz = np.array([[np.cos(theta), -1 * np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    result_val = np.matmul(rz, original_pos)

    x = result_val[0, 0]
    y = result_val[1, 0]
    z = result_val[2, 0]
  
    retval = [x, y, z]

    return retval

#Reads a magnetic test file
#fname = the file name and path for the file to parse
#scaled = flag to determine if the output data needs to be rescaled from nano tesla to tesla. If true data output is multiplie by 10^-9
def parse_mtf(fname, scaled=True):
    mtf_file = open(fname, "r")
    
    mtf_file.readline()

    radials = int(mtf_file.readline())
  
    delta_angle = int(360 / radials)
   
    mtf_file.readline()
    d1 = float(mtf_file.readline())
    mtf_file.readline()
    d2 = float(mtf_file.readline())
    mtf_file.readline()
    d3 = float(mtf_file.readline())
    mtf_file.readline()
    d4 = float(mtf_file.readline())

    positions = []

    #Create array of positions that all the measurements
    #were taken at. Convert from polar coordinates to cartesian
    for i in range(0, 360, delta_angle):

        positions.append(rotate(d1, 0, 0, i))
        positions.append(rotate(d2, 0, 0, i))
        positions.append(rotate(d3, 0, 0, i))
        positions.append(rotate(d4, 0, 0, i))

    values = []

    nano = 1
    if scaled:
        nano = 10**(-9)

    #Read in all of the measured data. We need to rotate data according to the angle it was recorded
    #at. This is also due to conversion from polar to cartesian coordinates
    for i in range(0, radials):
        mtf_file.readline()
        mtf_file.readline()
    
        x = float(mtf_file.readline())*nano
        y = float(mtf_file.readline())*nano
        z = float(mtf_file.readline())*nano

        values.append(rotate(x, y, z, 360-(i*delta_angle)))
        #values.append([x, y, z])

        x = float(mtf_file.readline())*nano
        y = float(mtf_file.readline())*nano
        z = float(mtf_file.readline())*nano
        values.append(rotate(x, y, z, 360-(i*delta_angle)))
        #values.append([x, y, z])

        x = float(mtf_file.readline())*nano
        y = float(mtf_file.readline())*nano
        z = float(mtf_file.readline())*nano
        values.append(rotate(x, y, z, 360-(i*delta_angle)))
        #values.append([x, y, z])

        x = float(mtf_file.readline())*nano
        y = float(mtf_file.readline())*nano
        z = float(mtf_file.readline())*nano
        values.append(rotate(x, y, z, 360-(i*delta_angle)))
        #values.append([x, y, z])
        
    #Pull the spherical harmonic calculated values
    mtf_file.readline()

    mtf_file.readline()
    x = float(mtf_file.readline())
   
    mtf_file.readline()
    y = float(mtf_file.readline())
    
    mtf_file.readline()
    z = float(mtf_file.readline())

    dipole = [x, y, z]
    
    mtf_file.close()

    return positions, values, dipole


class bf_simulator:
  def __init__(self, distances, delta, scale=1):
    self.distances = distances
    self.angle_delta = delta
    self.poles = 0
    self.scale = scale

    #Create the observer distance array
    self.dist_array = []
    for angle in range(0, 360, self.angle_delta):
      for pos in distances:   
        w = self.generate_position(pos, 0, 0, angle)
        self.dist_array.append(w)

    # Create the bfield matrix
    self.field_array = []    
    for positions in self.dist_array:
      w = self.get_bfield(positions[0], positions[1], positions[2], 0, 0, 0, 0, 0, 0)
      self.field_array.append(w)

    # Generate column vectors of distances
    self.x_pos = np.array([row[0] for row in self.dist_array])
    self.y_pos = np.array([row[1] for row in self.dist_array])
    self.z_pos = np.array([row[2] for row in self.dist_array])

    # Column vectors that will hold values of b-field
    self.x_val = []    
    self.y_val = []    
    self.z_val = []  

  # Create a vector of observer positions
  def generate_position(self, distx, disty, distz, theta):  
    theta = np.radians(theta)
      
    original_pos = np.array([[distx, disty, distz]]).T  
    rz = np.array([[np.cos(theta), -1 * np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    result_val = np.matmul(rz, original_pos)

    x = result_val[0, 0]
    y = result_val[1, 0]
    z = result_val[2, 0]
  
    retval = [x, y, z]

    return retval
  
  # add a bfield due to a dipole moment to the total bfield
  def add_dipole(self, mlocation_x, mlocation_y, mlocation_z, mval_x, mval_y, mval_z):
    #increment the number of poles
    self.poles = self.poles + 1

    new_field = []
    for positions in self.dist_array:
      new_field.append(self.get_bfield(positions[0], positions[1], positions[2], mval_x, mval_y, mval_z, mlocation_x, mlocation_y, mlocation_z))      

    self.field_array = np.add(self.field_array, new_field)
    self.x_val = [row[0] for row in self.field_array]
    self.y_val = [row[1] for row in self.field_array]
    self.z_val = [row[2] for row in self.field_array]

  # Helper function to generate a b-field at the position of the observer for a given
  # dipole moment at a specific location. Units are meters & amp-m^2
  def get_bfield(self, obs_x, obs_y, obs_z, mx, my, mz, mpos_x, mpos_y, mpos_z):

    mu_0 = (4.0 * np.pi) * 10.0**(-7.0)
    mu_denom = (4.0 * np.pi)

    L = mx * (obs_x - mpos_x) + my *(obs_y - mpos_y) + mz *(obs_z - mpos_z)
    P = np.sqrt((obs_x - mpos_x)**2 + (obs_y - mpos_y)**2 + (obs_z - mpos_z)**2)

    #Check for a divide by zero
    if P == 0:
      #set it to a super small number if it is zero
      P = 1e-16

    Bx = (mu_0/mu_denom) * ((3*(obs_x - mpos_x))/P**5) * L - (mu_0/mu_denom) * (mx/P**3)
    Bx = Bx * self.scale
    By = (mu_0/mu_denom) * ((3*(obs_y - mpos_y))/P**5) * L - (mu_0/mu_denom) * (my/P**3)
    By = By * self.scale
    Bz = (mu_0/mu_denom) * ((3*(obs_z - mpos_z))/P**5) * L - (mu_0/mu_denom) * (mz/P**3)
    Bz = Bz * self.scale

    return [Bx, By, Bz]

  # Create a contour plot of the b-field that has been calculated
  def plot(self, axis='X', plevels=20):
    
    ngridx = 200
    ngridy = 200
    
    x = self.x_pos
    y = self.y_pos

    if axis=='X':
      z = (x * 0) + self.x_val
    if axis=='Y':
      z = (x * 0) + self.y_val
    if axis=='Z':
      z = (x * 0) + self.z_val

    fig, ax2= plt.subplots(nrows=1)

    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.

    # Create grid values first.
    xi = np.linspace(-2.1, 2.1, ngridx)
    yi = np.linspace(-2.1, 2.1, ngridy)

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')



    # ----------
    # Tricontour
    # ----------
    # Directly supply the unordered, irregularly spaced coordinates
    # to tricontour.

    ax2.tricontour(x, y, z, levels=plevels, linewidths=0.5, colors='k')
    cntr2 = ax2.tricontourf(x, y, z, levels=plevels, cmap="plasma")

    strDisplay = axis + "-Axis, Poles: %d"

    fig.colorbar(cntr2, ax=ax2)
    ax2.plot(x, y, 'ko', ms=3)
    ax2.set(xlim=(-2, 2), ylim=(-2, 2))
    ax2.set_title(strDisplay % self.poles)

    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

  # Get the ordered data for training purposes
  def get_data(self):
    return self.dist_array, self.field_array