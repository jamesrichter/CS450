import random
import numpy as np
import math

LEARNING_RATE = 0.01

def sigmoid(x):
   """A sigmoid function with a gradual slope."""
   if x > 100:
      return 1
   if x < -100:
      return 0
   return 1.0 / (1.0 + math.exp(-4.9*x))

class Neuron:
   """A single neuron in our neural network."""
   def __init__(self, width):
      self.incoming_weight = [random.random()-.5 for x in range(width)]
      self.value = 0
   def __repr__(self):
      neuronString = str(self.incoming_weight) + '\\' + str(self.value)
      return neuronString

class Network:
   """
   A neural network, used for calculating outputs to the system.
   """
   def categorizeOutput(self):
      i = 0
      self.targets = list(self.targets)
      new_target = [0 for x in range(self.num_outputs)]
      for target in self.targets:
         new_target[target] = 1
         self.targets[i] = new_target
         new_target = [0 for x in range(self.num_outputs)]
         i+=1

   #calculate the error of the output nodes
   def outputError(self, index):
      self.hidden_layers = np.array(self.hidden_layers)
      outputs = []
      for node in self.output_layer:
         outputs.append(node.value)
      print outputs
      target = self.targets[index]
      target = np.array(target)
      outputs = np.array(outputs)
      print target
      error = (outputs - target) * outputs * (1 - outputs)
      self.output_error = error

   def hiddenError(self, index):
      output_error = self.output_error
      for layer in reversed(hidden_layers):
         for node in layer:
            error = 5
         output_error = error
      
      
      weight_matrix = []
      for node in self.hidden_layers[:,-1]:
         weight_matrix.append(node.incoming_weight)
      error_matrix = np.array(weight_matrix) * np.array(self.output_error).transpose() 
      error_matrix = error_matrix
      self.hidden_error = error

   def updateWeights(self):
      pass

   def __init__(self, data, target, num_outputs, layers):
      self.layers = layers
      self.data = data
      self.num_outputs = num_outputs
      self.targets = target
      self.width = 0
      self.hidden_layers = []
      self.input_layer = []
      self.output_layer = []
      self.generateNetwork()

   def trainNetwork(self):
      self.categorizeOutput()
      i = 0
      for data_pt in self.data:
         self.activateNetwork(data_pt)
         self.outputError(i)
         self.hiddenError(i)
         i+=1

   def generateNetwork(self):
      self.input_layer = [Neuron(0) for count in range(self.width + 1)]
      self.hidden_layers = []
      i = 0
      for x in self.layers:
         if i == 0:
            self.hidden_layers.append([Neuron(self.width + 1) for count in range(x + 1)])
         else:
            self.hidden_layers.append([Neuron(self.layers[i-1] + 1) for count in range(x + 1)])
         i += 1
      self.output_layer = [Neuron(self.layers[-1]+1) for count in range(self.num_outputs)]

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
         x[-1].incoming_weight = []
         layer += 1
      for neuron in self.output_layer:
         i = 0
         for weight in neuron.incoming_weight:
            neuron.value += weight * self.hidden_layers[-1][i].value
            i+=1
         neuron.value = sigmoid(neuron.value)

      print "done :)"