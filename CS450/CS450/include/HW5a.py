import random

def sigmoid(x):
   """A sigmoid function with a gradual slope."""
   if x > 0:
      return 1
   else:
      return 0

   #if x > 100:
   #   return 1
   #if x < -100:
   #   return 0
   #return 2.0 / (1.0 + math.exp(-4.9*x))

class Neuron:
   """A single neuron in our neural network."""
   def __init__(self, width):
      self.incoming_weight = [random.random()-.5 for x in range(width)]
      self.value = 0
   def __repr__(self):
      neuronString = str(self.outgoing) + '\\' + str(self.value)
      return neuronString

class Network:
   """
   A neural network, used for calculating outputs to the system.
   """
   def __init__(self, data):
      self.layers = 0
      self.data = data
      self.width = 0
      self.hidden_layers = []
      self.input_layer = []
      self.generateNetwork()

   def generateNetwork(self):
      self.input_layer = [Neuron(0) for count in range(self.width + 1)]
      self.hidden_layers = []
      for x in range(self.layers):
         self.hidden_layers.append([Neuron(self.width + 1) for count in range(self.width + 1)])

   def activateNetwork(self, inputs):
      """
      Activate the neural network.  inputs should not include the bias.
      """
      #outputs = {}
      assert self.input_layer.__len__() == inputs.__len__() + 1

      # plug in the inputs
      for x in range(inputs.__len__()):
         self.input_layer[x].value = inputs[x]
      
      self.input_layer[-1].value = 1

      # calculate all of the layers
      layer = 0
      for x in self.hidden_layers:
         for neuron in x:
            if layer == 0:
               i = 0
               for weight in neuron.incoming_weight:
                  neuron.value += weight * self.input_layer[i].value
                  i += 1
               neuron.value = sigmoid(neuron.value)
            else:
               i = 0
               for weight in neuron.incoming_weight:
                  neuron.value += weight * self.hidden_layers[layer - 1][i].value
                  i += 1
               neuron.value = sigmoid(neuron.value)
         x[-1].value = 1
         layer += 1
      print "done :)"