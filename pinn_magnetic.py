from logging import logProcesses
import tensorflow as tf
import numpy as np
import keras
from statistics import mean


print("TensorFlow version:", tf.__version__)

class DipoleRegressor:
  def __init__(self, num_poles, l_rate=0.1, o_scaled=True):
    #True if the training data comes in with values on the order of 10^-9
    self.scaled = o_scaled
    w_shape = (num_poles, 3)
    w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
    self.w = tf.Variable(w_initial_value, trainable=True)
    self.learning_rate = l_rate
    # The model will use stochastic gradient descent
    self.opt = tf.keras.optimizers.SGD(learning_rate=l_rate)
    self.loss_object = tf.keras.losses.MeanSquaredError()    

  def __call__(self, inputs):
    
    # Make sure the inputs are in a tensor format
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    # Define P
    p = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs)), name="define P")    
    #Define L
    l = tf.math.reduce_sum(tf.math.multiply(self.w, inputs))
    # Define base mu terms
    mu_term = tf.constant(((4.0 * np.pi) * 10.0**(-7.0)) / (4.0 * np.pi))
    mu3 = tf.constant(tf.math.scalar_mul(3.0, mu_term))
    
    # Calculate the B-Field at the supplied position (inputs)
    retval = tf.math.scalar_mul( l, tf.math.scalar_mul( mu3 , tf.math.scalar_mul(tf.math.divide(1.0, tf.math.pow(p, 5.0)), inputs)))- tf.math.scalar_mul( mu_term, tf.math.scalar_mul(tf.math.divide(1.0, tf.math.pow(p,3.0)), self.w))
    
    return tf.math.reduce_sum(retval, 0)
    
  @property
  def weights(self):
    return [self.w]

  def dipole(self):
    return tf.math.reduce_sum(self.w, 0)

  def train(self, x, y):
    
    with tf.GradientTape() as tape:      
      # Determine the loss dx/dy while recording math operations
      loss = self.loss_object(self(x), y)
    
    gradients = tape.gradient(loss, self.w)
    self.update_weights(gradients, self.w)
    scaled_loss = loss

    #See if we need to scale the losses
    if self.scaled:
        scaled_loss = tf.math.scalar_mul( 1.0e9, loss)

    return scaled_loss

  def update_weights(self, gradients, weights):
    
    # rescale gradients from nano-tesla range to am^2 range
    rscale = tf.Variable(tf.math.scalar_mul(1.0e9, gradients))

    # apply learning rate
    delt = tf.Variable(tf.math.scalar_mul(self.learning_rate, rscale))
    
    self.w.assign_sub(delt)
    

class DipoleModel(tf.keras.Model):
  def __init__(self, lrate=1000, *args, **kwargs):
    super(DipoleModel, self).__init__(*args, **kwargs)
    self.loss_object = tf.keras.losses.MeanSquaredError()
    self.loss_object.reduction="none"
    self.layer = DipoleLayer(lrate=lrate)
    self.loss_history = []
    

  def call(self, inputs):
    return self.layer(inputs)

  
  def train_step(self, position, value):
    x = position 
    y_true = value
    with tf.GradientTape() as tape:      
      y_pred = self(x, training=True)
      # Determine the loss dx/dy while recording math operations      
      loss = self.loss_object(y_true=y_true, y_pred=y_pred)
    
    
    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    #self.layer.update_weights(gradients)
    
    #print(trainable_vars)
    grads = tf.convert_to_tensor(gradients, dtype=tf.float32)
    scaled_grads = tf.Variable(tf.math.scalar_mul(1.0e9, grads))
    
    #update weights    
    self.layer.update_weights(scaled_grads)
    

    scaled_loss = tf.math.scalar_mul( 1.0e9, loss)
    return scaled_loss

  def fit(self, positions, values, epochs):
    
    # Clear the old learning history
    self.loss_history = []

    for i in range(epochs): 
      loss_epoch = []     
      for x in range(len(values)):
        loss_epoch.append(self.train_step(positions[x], values[x]).numpy())
      
      #Only take the mean of the epoch for the loss history
      self.loss_history.append(mean(loss_epoch))
      print("epoch ", i, "--- Loss ---", self.loss_history[-1])

  
  def dipole(self):
    return tf.math.reduce_sum(self.layer.trainable_weights, 0).numpy()
  


class MultiDipoleModel(tf.keras.Model):
  def __init__(self, poles=1, lrate=1000, *args, **kwargs):
    super(MultiDipoleModel, self).__init__(*args, **kwargs)
    self.loss_object = tf.keras.losses.MeanSquaredError()
    self.loss_object.reduction="none"
    self.DipoleLayers = []
    self.loss_history = []
    self.num_poles = poles
    
    for i in range(self.num_poles):
      self.DipoleLayers.append(DipoleLayer(lrate=lrate))

  def call(self, inputs):    
    predictions = []

    #gather predictions
    for i in range(self.num_poles):
      x = self.DipoleLayers[i](inputs)
      #print(x)
      predictions.append(tf.reshape(x, [1,3]))
    
    #Sum the preditions - We can probably use a better way to do this. Something like tf.keras.layers.add([x1, x2])
    concat_predictions = keras.layers.concatenate(predictions, axis=0)
    y = tf.reduce_sum(concat_predictions, axis=0, keepdims=True)
    y = tf.reshape(y, [3,])
    
    return y
      
  def train_step(self, position, value):
    x = position 
    y_true = value
    #Record the prediction operations on the gradient tape
    with tf.GradientTape(persistent=True) as tape:      
      y_pred = self(x, training=True)
      # Determine the loss dx/dy while recording math operations      
      loss = self.loss_object(y_true=y_true, y_pred=y_pred)
    
    #Determine gradients and perform back propogation
    for i in range(self.num_poles):
      grad = tape.gradient(loss, self.DipoleLayers[i].w)
      scaled_grad = tf.Variable(tf.math.scalar_mul(1.0e9, grad))
      self.DipoleLayers[i].update_weights(scaled_grad)

    #delete the gradient tape
    del tape
    
    #Rescale the losses with respect to a 1 nT resolution
    scaled_loss = tf.math.scalar_mul( 1.0e9, loss)
    return scaled_loss

  def fit(self, positions, values, epochs):
    
    # Clear the old learning history
    self.loss_history = []

    for i in range(epochs): 
      loss_epoch = []     
      for x in range(len(values)):
        loss_epoch.append(self.train_step(positions[x], values[x]).numpy())
      
      #Only take the mean of the epoch for the loss history
      self.loss_history.append(mean(loss_epoch))
      print("epoch ", i, "--- Loss ---", self.loss_history[-1])

  
  def dipole(self):
    return tf.math.reduce_sum(self.trainable_variables, 0).numpy()
  



class DipoleLayer(keras.layers.Layer):
  def __init__(self, units=3, input_dim=3, lrate=1000):
    super(DipoleLayer, self).__init__()
    w_shape = (0, 3)
    w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
    self.w = self.add_weight(shape=(1, 3), initializer="random_normal", trainable=True)
    #self.w = tf.Variable(w_initial_value, trainable=True)
    self.learning_rate = lrate

  def call(self, inputs):
    # Make sure the inputs are in a tensor format
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    # Define P
    p = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs)), name="define P")    
    #Define L
    l = tf.math.reduce_sum(tf.math.multiply(self.w, inputs))
    # Define base mu terms
    mu_term = tf.constant(((4.0 * np.pi) * 10.0**(-7.0)) / (4.0 * np.pi))
    mu3 = tf.constant(tf.math.scalar_mul(3.0, mu_term))
    
    # Calculate the B-Field at the supplied position (inputs)
    retval = tf.math.scalar_mul( l, tf.math.scalar_mul( mu3 , tf.math.scalar_mul(tf.math.divide(1.0, tf.math.pow(p, 5.0)), inputs)))- tf.math.scalar_mul( mu_term, tf.math.scalar_mul(tf.math.divide(1.0, tf.math.pow(p,3.0)), self.w))
    
    return tf.math.reduce_sum(retval, 0)
  
  def update_weights(self, gradients):

    # apply learning rate
    delt = tf.Variable(tf.math.scalar_mul(self.learning_rate, gradients))
    
    # make sure the tensor is the same  shape as the weights
    delt = tf.reshape(delt, [1, 3])
    
    self.w.assign_sub(delt)
    
