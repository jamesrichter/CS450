import random
import numpy as np
import math

LEARNING_RATE = .9
random.seed(5)

def sigmoid(x):
   """A sigmoid function with a gradual slope."""
   if x > 100:
      return 1
   if x < -100:
      return 0
   return 1.0 / (1.0 + math.exp(-x))

class Neuron:
   """A single neuron in our neural network."""
   def __init__(self, width):
      self.incoming_weight = [(random.random()-.5)/100 for x in range(width)]
      self.value = 0
   def __repr__(self):
      neuronString = str(self.incoming_weight) + '\\' + str(self.value)
      return neuronString

class Network:
   """
   A neural network, used for calculating outputs to the system.
   """
   # change 3 to [0 0 0 1]
   def categorizeOutput(self):
      i = 0
      self.targets = list(self.targets)
      new_target = [0 for x in range(self.num_outputs)]
      for target in self.targets:
         new_target[int(target)] = 1
         self.targets[i] = new_target
         new_target = [0 for x in range(self.num_outputs)]
         i+=1

   #calculate the error of the output nodes
   def outputError(self, index):
      self.hidden_layers = np.array(self.hidden_layers)
      i = 0
      for node in self.output_layer:
         target = self.targets[index][i]
         aj = node.value
         delta = (aj - target) * aj * (1 - aj)
         node.output_delta = delta
         node.new_weights = []
         j = 0
         for weight in node.incoming_weight:
            new_weight = weight - LEARNING_RATE * delta * self.hidden_layers[-1][j].value
            node.new_weights.append(new_weight)
            j+=1
         i += 1

   def hiddenError(self, index):
      self.hidden_layers = np.array(self.hidden_layers)
      i = 0
      for layer in reversed(self.hidden_layers):
         j = 0
         for node in layer:
            addend = 0
            aj = node.value
            if i == 0:
               for node2 in self.output_layer:
                  addend += node2.incoming_weight[j] * node2.output_delta
            else:
               for node2 in (self.hidden_layers[::-1])[i-1][:-1]:
                  addend += node2.incoming_weight[j] * node2.delta
            j+=1
            node.delta = aj*(1-aj)*addend
            node.new_weights = []
            k = 0
            for weight in node.incoming_weight:
               try:
                  node.new_weights.append(weight - node.delta*LEARNING_RATE*(self.hidden_layers[::-1])[i+1][k].value)
               except:
                  node.new_weights.append(weight - node.delta*LEARNING_RATE*self.input_layer[k].value)
               k +=1
         i+=1

   def updateWeights(self):
      for layer in self.hidden_layers:
         for node in layer:
            node.incoming_weight = node.new_weights
      for node in self.output_layer:
         node.incoming_weight = node.new_weights

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

   def __repr__(self):
      ret_string = ""
      ret_string+="Layers: "
      ret_string+=" " + str(self.input_layer.__len__())
      for x in self.hidden_layers:
         ret_string+=" " + str(x.__len__())
      ret_string+=" " + str(self.output_layer.__len__())
      ret_string+="\nWeights: "
      for x in self.hidden_layers:
         for node in x:
            for weight in node.incoming_weight:
               ret_string+=str(weight) + " "
            ret_string+='\n'
         ret_string+='\n'
      for node in self.output_layer:
         for weight in node.incoming_weight:
            ret_string+=str(weight) + " "
         ret_string+='\n'
      return ret_string

   def trainNetwork(self):
      i = 0
      for data_pt in self.data:
         self.activateNetwork(data_pt)
         self.outputError(i)
         self.hiddenError(i)
         self.updateWeights()
         i+=1

   def testNetwork(self, test_data, test_target):
      i = 0
      num_right = 0
      for datapt in test_data:
         self.activateNetwork(datapt)
         values = []
         for node in self.output_layer:
            values.append(node.value)
         if int(values.index(max(values))) == int(test_target[i]):
            num_right +=1
         i+=1

      return num_right/float(i)

   def generateNetwork(self):
      self.categorizeOutput()
      self.input_layer = [Neuron(0) for count in range(self.data[0].__len__() + 1)]
      self.hidden_layers = []
      i = 0
      for x in self.layers:
         if i == 0:
            self.hidden_layers.append([Neuron(self.data[0].__len__() + 1) for count in range(x + 1)])
         else:
            self.hidden_layers.append([Neuron(self.layers[i-1] + 1) for count in range(x + 1)])
         i += 1
      self.output_layer = [Neuron(self.layers[-1]+1) for count in range(self.num_outputs)]

   def flushNetwork(self):
      for layer in self.hidden_layers:
         for node in layer:
            node.value = 0
         layer[-1].value = 1
            
      for node in self.output_layer:
         node.value = 0

   def activateNetwork(self, inputs):
      """
      Activate the neural network.  inputs should not include the bias.
      """
      #outputs = {}
      assert self.input_layer.__len__() == inputs.__len__() + 1
      self.flushNetwork()

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

      #print "done :)"