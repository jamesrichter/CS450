"""
network.py
Author: James Richter
Class: CS 499, Twitchell/Burton
Last Updated: 1/6/2016

The neural network is the "brain."
"""
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
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

nn = Network(train_data, train_target, 3, [6,7,4,3])
width = train_data[0].__len__()
nn.width = width
nn.generateNetwork()
nn.activateNetwork(nn.data[0])
nn.trainNetwork()


array = np.loadtxt('./data/pima-indians-diabetes.data', dtype = "U", delimiter=',')
house_data = array[:,1::]
house_target = array[:,0]
house_data = normalize(house_data, axis=1)
train_data, test_data, train_target, test_target = train_test_split(house_data, house_target, test_size=0.30, random_state=2)
nn = Network(train_data, train_target, 3, [4,5,4,2])
width = train_data[0].__len__()
nn.width = width
nn.generateNetwork()
nn.activateNetwork(nn.data[0])