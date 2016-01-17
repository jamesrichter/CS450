import random
from sklearn import datasets

class DataPoint():
   # the almighty datapoint
   def __init__(self, data1, target1):
      self.data = data1
      self.target = target1
      self.train = None

   def __repr__(self):
      return str(self.data) + ' ' + str(self.target)

class HardCoded():
   def train(self, training_set):
      pass

   def predict(self, instances):
      return 0

class DataSet():
   def __init__(self):
      self.data_points = None

   def __repr__(self):
      return str(self.data_points)

   def loadData(self, filename):
      read_file1 = open(filename, 'r+')
      i = 0
      data = {}
      for line in read_file1:
         array1 = line.strip().split(',')
         data[i] = DataPoint(array1[0:-1], array1[-1])
         i += 1
      self.data_points = data

iris = datasets.load_iris() 

assert iris.data.__len__() == iris.target.__len__()

data_len = iris.data.__len__()
iris_data_points = {}

for i in range(data_len):
   iris_data_points[i] = DataPoint(iris.data[i], iris.target[i])

keys = random.sample(iris_data_points, data_len)
training = keys[0:data_len*7/10]
test = keys[data_len*7/10:data_len]

classifier = HardCoded()
classifier.train(training)
num_right = 0
num_total = test.__len__()
for i in range(num_total):
   if classifier.predict(iris_data_points[test[i]]) == iris_data_points[test[i]].target:
      num_right += 1

print num_right/float(num_total)
print ":)"
car_data_points = DataSet()
car_data_points.loadData('./data/car.data')
print car_data_points

