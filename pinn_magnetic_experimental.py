from logging import logProcesses
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from statistics import mean



print("TensorFlow version:", tf.__version__)

class DipoleModel(tf.keras.Model):
  def __init__(self, lrate=1000, optimizer='sgd', loss='mse', scale=1, *args, **kwargs):
    super(DipoleModel, self).__init__(*args, **kwargs)
    
    self.layer = DipoleLayer(scale=scale)
    self.loss_history = []
    
    #Setup the optimizer
    match optimizer:
      case 'sgd':
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lrate)
      case 'rmsprop':
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrate)
      case 'adam':
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
      case 'nadam':
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
      case 'adadelta':
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lrate)
      case 'adagrad':
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lrate)
    
    #Setup the loss function
    match loss:
      case 'mse':
        self.loss_object = tf.keras.losses.MeanSquaredError()
      case 'mae':
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
      case 'huber':
        self.loss_object = tf.keras.losses.Huber()

    self.loss_object.reduction="none"
    

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
    gradients = tape.gradient(loss, self.trainable_variables)    
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))



    #normalize the losses into the nano-tesla range
    #scaled_loss = tf.math.scalar_mul( 1.0e9, loss)
    #return scaled_loss
    return loss

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
  
class DipoleLayer(keras.layers.Layer):
  def __init__(self, units=3, input_dim=3, scale=1):
    super(DipoleLayer, self).__init__()
    w_shape = (0, 3)
    self.w = self.add_weight(shape=(1, 3), initializer="random_normal", trainable=True)
    self.scale = scale
  
  def call(self, inputs):
    # Make sure the inputs are in a tensor format
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    # Define P
    p = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs)), name="define P")    
    #Define L
    l = tf.math.reduce_sum(tf.math.multiply(self.w, inputs))
    # Define base mu terms
    #mu_term = tf.constant(((4.0 * np.pi) * 10.0**(-7.0)) / (4.0 * np.pi))
    mu_term = tf.constant((1.256637061e-6) / (4.0 * np.pi))
    mu3 = tf.constant(tf.math.scalar_mul(3.0, mu_term))
    
    # Calculate the B-Field at the supplied position (inputs)
    retval = tf.math.scalar_mul( l, tf.math.scalar_mul( mu3 , tf.math.scalar_mul(tf.math.divide(1.0, tf.math.pow(p, 5.0)), inputs)))- tf.math.scalar_mul( mu_term, tf.math.scalar_mul(tf.math.divide(1.0, tf.math.pow(p,3.0)), self.w))
    
    # Scale the output to the designed output scale
    retval = tf.math.scalar_mul(self.scale, retval)

    return tf.math.reduce_sum(retval, 0)
  
  


class MultiDipoleModel(tf.keras.Model):
  def __init__(self, poles=1, lrate=1000, optimizer='adam', loss='mse', scale=1, early_stop=False, target_stop=1, *args, **kwargs):
    super(MultiDipoleModel, self).__init__(*args, **kwargs)
    self.loss_object = tf.keras.losses.MeanSquaredError()
    self.loss_object.reduction="none"
    self.DipoleLayers = []
    self.loss_history = []
    self.num_poles = poles
    self.early_stop = early_stop
    self.target_stop = target_stop
    
    for i in range(self.num_poles):
      self.DipoleLayers.append(DipoleLayer(scale=scale))

    
    #Setup the optimizer
    match optimizer:
      case 'sgd':
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lrate)
      case 'rmsprop':
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrate)
      case 'adam':
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
      case 'nadam':
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
      case 'adadelta':
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lrate)
      case 'adagrad':
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lrate)
    
    #Setup the loss function
    match loss:
      case 'mse':
        self.loss_object = tf.keras.losses.MeanSquaredError()
      case 'mae':
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
      case 'huber':
        self.loss_object = tf.keras.losses.Huber()

    self.loss_object.reduction="none"

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
    with tf.GradientTape() as tape:      
      y_pred = self(x, training=True)
      # Determine the loss dx/dy while recording math operations      
      loss = self.loss_object(y_true=y_true, y_pred=y_pred)
    
    # Compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)    
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    return loss

  def fit(self, positions, values, epochs):
    
    # Clear the old learning history
    self.loss_history = []

    for i in range(epochs): 
      loss_epoch = []     
      for x in range(len(values)):
        loss_epoch.append(self.train_step(positions[x], values[x]).numpy())
            
      # Make a copy of the last epoch loss for comparison
      if len(self.loss_history) > 0:
        old_loss = self.loss_history[-1]
      else:
        old_loss = mean(loss_epoch)
        
      #Only take the mean of the epoch for the loss history
      self.loss_history.append(mean(loss_epoch))
      print("epoch ", i, "--- Loss ---", self.loss_history[-1])
      
       #If the loss is in the range of the early stoppage target then kill the training
      if self.loss_history[-1] <= self.target_stop:
        break
      # check for an early stoppage for increased error
      if (self.early_stop == True) and (self.loss_history[-1] > old_loss):        
        break
      

  
  def dipole(self):
    return tf.math.reduce_sum(self.trainable_variables, 0).numpy()

class MultiPoleModel(tf.keras.Model):
  def __init__(self, moments=1, lrate=.01, krate=.001, optimizer='adam', loss='mse', scale=1, early_stop=False, target_stop=1, *args, **kwargs):
    super(MultiPoleModel, self).__init__(*args, **kwargs)
    self.loss_object = tf.keras.losses.MeanSquaredError()
    self.loss_object.reduction="none"
    self.MomentLayers = []
    self.loss_history = []
    self.num_moments = moments
    self.early_stop = early_stop
    self.target_stop = target_stop
    
    for i in range(self.num_moments):
      self.MomentLayers.append(MagneticMomentLayer(scale=scale))

      #Setup the optimizer
    match optimizer:
      case 'sgd':
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lrate)
      case 'rmsprop':
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrate)
      case 'adam':
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
      case 'nadam':
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
      case 'adadelta':
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lrate)
      case 'adagrad':
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lrate)
    
    #Setup the loss function
    match loss:
      case 'mse':
        self.loss_object = tf.keras.losses.MeanSquaredError()
      case 'mae':
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
      case 'huber':
        self.loss_object = tf.keras.losses.Huber()

    self.loss_object.reduction="none"

  def call(self, inputs):    
    predictions = []

    #gather predictions
    for i in range(self.num_moments):
      x = self.MomentLayers[i](inputs)
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
    with tf.GradientTape() as tape:      
      y_pred = self(x, training=True)
      # Determine the loss dx/dy while recording math operations      
      loss = self.loss_object(y_true=y_true, y_pred=y_pred)
    
    # Compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)    
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return loss

  def fit(self, positions, values, epochs):
    
    # Clear the old learning history
    self.loss_history = []

    for i in range(epochs): 
      loss_epoch = []     
      for x in range(len(values)):        
        loss_epoch.append(self.train_step(positions[x], values[x]).numpy())
        #print(loss_epoch[-1])

      # Make a copy of the last epoch loss for comparison
      if len(self.loss_history) > 0:
        old_loss = self.loss_history[-1]
      else:
        old_loss = mean(loss_epoch)
      
      #Only take the mean of the epoch for the loss history
      self.loss_history.append(mean(loss_epoch))
      print("epoch ", i, "--- Loss ---", self.loss_history[-1])

      #If the loss is in the range of the early stoppage target then kill the training
      if self.loss_history[-1] <= self.target_stop:
        break
      # check for an early stoppage for increased error
      if (self.early_stop == True) and (self.loss_history[-1] > old_loss):        
        break

  
  def dipole(self):
    return tf.math.reduce_sum(self.trainable_variables, 0).numpy()
  
  def moment(self):
    
    for i in range(self.num_moments):
      print("---- Moment ", i, "-----")
      print("Position: ", self.MomentLayers[i].k.numpy())
      print("Value:    ", self.MomentLayers[i].w.numpy())
      #retval.append([self.MomentLayers[i].k.numpy(), self.MomentLayers[i].w.numpy()])
    
  
  
class MagneticMomentLayer(keras.layers.Layer):
  def __init__(self, units=3, input_dim=3, scale=1):
    super(MagneticMomentLayer, self).__init__()
    w_shape = (0, 3)
    w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
    self.w = self.add_weight(shape=(1, 3), initializer="random_normal", trainable=True)
    #initer = tf.keras.initializers.Zeros()
    self.k = self.add_weight(shape=(1, 3), initializer="random_normal", trainable=True)
    
    self.scale = scale

  def call(self, inputs):
    # Make sure the inputs are in a tensor format
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    #calculate the difference between the observer position and the moment position
    delta_pos = tf.math.subtract(inputs, self.k)
    #print("delta_pos:", delta_pos)
    # Define P
    p = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(delta_pos)), name="define P")    
    #print("P: ", p)
    #Define L
    l = tf.math.reduce_sum(tf.math.multiply(self.w, delta_pos))
    #print("L: ", l)
    # Define base mu terms
    #mu_term = tf.constant(((4.0 * np.pi) * 10.0**(-7.0)) / (4.0 * np.pi))
    mu_term = tf.constant((1.256637061e-6) / (4.0 * np.pi))
    
    numerator1 = tf.math.scalar_mul(3, delta_pos)
    denom1 = tf.math.pow(p, 5.0)

    denom2 = tf.math.pow(p, 3.0)

    term1 = tf.math.scalar_mul(l, tf.math.scalar_mul(mu_term, tf.math.divide_no_nan(numerator1, denom1)))

    term2 = tf.math.scalar_mul(mu_term, tf.math.divide_no_nan(self.w, denom2))

    retval = tf.math.subtract(term1, term2)

    # Scale the output to the designed output scale
    retval = tf.math.scalar_mul(self.scale, retval)
    
    
    return tf.math.reduce_sum(retval, 0)



