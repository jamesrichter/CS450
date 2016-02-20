"""
network.py
Author: James Richter
Class: CS 499, Twitchell/Burton
Last Updated: 1/6/2016

The neural network is the "brain."
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn import tree
import math
import argparse
import numpy as np
from include.HW5a import Network

parser = argparse.ArgumentParser(description='Process some layers.')
parser.add_argument('layers', metavar='N', type=int, nargs=1,
                   help='the number of layers in our NN')

args = parser.parse_args()
#print args.layers[0]


iris = load_iris()
iris.data = normalize(iris.data, axis=0)
train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.30, random_state=2)
nn = Network(train_data, train_target, 3, [3])
nn.activateNetwork(nn.data[0])
s = []
t = []
tries = 0
i = 0
while i < 10000:
   print i,
   nn.trainNetwork()
   t.append(nn.testNetwork(test_data, test_target))
   if i > 100 and t[-1] <= t[-2]:
      tries+=1
   else:
      tries = 0
   s.append(i)
   if tries > 20:
      i = 10001
   i += 1
plt.plot(s, t)
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
plt.title('Iris')
plt.grid(True)
plt.savefig("nn_iris.png")

#clf = MLPClassifier(algorithm='l-bfgs', alpha=.1, hidden_layer_sizes=(4), random_state=1)
#clf.fit(train_data, train_target) 
#i = 0
#for x in clf.predict(test_data):
#   num_right = 0
#   if x == test_target[i]:
#      num_right += 1
#   i +=1
#print "clf accuracy:", num_right / float(i)


array = np.loadtxt('./data/pima-indians-diabetes.data', dtype = "U", delimiter=',')
house_data = array[:,::-1]
house_target = array[:,-1]
house_data = normalize(house_data, axis=1)
train_data, test_data, train_target, test_target = train_test_split(house_data, house_target, test_size=0.30, random_state=2)
nn = Network(train_data, train_target, 2, [8])
nn.activateNetwork(nn.data[0])
plt.clf()
s = []
t = []
tries = 0
i = 0
while i < 10000:
   nn.trainNetwork()
   print i, 
   t.append(nn.testNetwork(test_data, test_target))
   if i > 100 and t[-1] <= t[-2]:
      tries+=1
   else:
      tries = 0
   s.append(i)
   if tries > 100:
      i = 10001
   i +=1 
plt.plot(s, t)
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
plt.title('PIMA')
plt.grid(True)
plt.savefig("nn_pima.png")
plt.show()

#clf = MLPClassifier(algorithm='l-bfgs', alpha=.1, hidden_layer_sizes=(8), random_state=1)
#clf.fit(train_data, train_target) 
#i = 0
#for x in clf.predict(test_data):
#   num_right = 0
#   if x == test_target[i]:
#      num_right += 1
#   i +=1
#print "clf accuracy:", num_right / float(i)