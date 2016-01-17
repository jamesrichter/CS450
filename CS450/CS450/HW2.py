import random
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

class DataPoint():
   # the almighty datapoint
   def __init__(self, data1, target1):
      self.data = data1
      self.target = target1
      self.train = None

   def __repr__(self):
      return str(self.data) + ' ' + str(self.target)

class DataSet():
   def __init__(self):
      self.data_points = {}

   def __repr__(self):
      return str(self.data_points)

   def loadData(self, filename1):
      read_file1 = open(filename1, 'r+')
      i = 0
      data = {}
      for line in read_file1:
         array1 = line.strip().split(',')
         data[i] = DataPoint(array1[0:-1], array1[-1])
         i += 1
      self.data_points = data

   def addDataPoint(self, data_point1):
      if not self.data_points:
         self.data_points[i] = data_point1
      else:
         max_index = max(self.data_points)
         self.data_points[max_index + 1] = data_point1

   def numpyArrayData():
      if not self.data_points:
         return []
      else:
         temp_array = []
         for point in self.data_points:
            temp_array.append(point.data)
         return np.array(temp_array)

   def numpyArrayTarget():
      if not self.data_points:
         return []
      else:
         temp_array = []
         for point in self.data_points:
            temp_array.append(point.target)
         return np.array(temp_array)

iris = datasets.load_iris() 
assert iris.data.__len__() == iris.target.__len__()
data_len = iris.data.__len__()
iris_data_points = DataSet()

for i in range(data_len):
   data_point = DataPoint(iris.data[i], iris.target[i])
   iris_data_points.addDataPoint(data_point)

keys = random.sample(iris_data_points, data_len)
training = keys[0:data_len*7/10]
test = keys[data_len*7/10:data_len]

num_right = 0
num_total = test.__len__()
for i in range(num_total):
   if 0 == iris_data_points[test[i]].target:
      num_right += 1

print num_right/float(num_total)

car_data_points = DataSet()
car_data_points.loadData('./data/car.data')
print car_data_points

classifier = KNeighborsClassifier(n_neighbors=3)
data1 = car_data_points.numpyArrayData()
target1 = car_data_points.numpyArrayTarget()
X_train, X_test, y_train, y_test = train_test_split(data1, target1, test_size=0.30, random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print X_train, X_test, y_train, y_test
print predictions