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

   def numpyArrayData(self):
      if not self.data_points:
         return []
      else:
         temp_array = []
         for _,point in self.data_points.items():
            temp_array.append(point.data)
         return np.array(temp_array)

   def numpyArrayTarget(self):
      if not self.data_points:
         return []
      else:
         temp_array = []
         for _,point in self.data_points.items():
            temp_array.append(point.target)
         return np.array(temp_array)

iris = datasets.load_iris() 
assert iris.data.__len__() == iris.target.__len__()
data_len = iris.data.__len__()
iris_data_points = DataSet()

for i in range(data_len):
   data_point = DataPoint(iris.data[i], iris.target[i])
   iris_data_points.addDataPoint(data_point)

read_file1 = open('./data/car.data', 'r+')
write_file1 = open('./data/car.data2', 'w+')
i = 0
data = read_file1.read()
data = data.replace("vhigh", '4')
data = data.replace("low", '1')
data = data.replace("high", '3')
data = data.replace("med", '2')
data = data.replace("big", '3')
data = data.replace("small", '1')
data = data.replace("5more", '5')
data = data.replace("more", '6')
data = data.replace("unacc", '1')
data = data.replace("acc", '2')
data = data.replace("vgood", '4')
data = data.replace("good", '3')
print data
write_file1.write(data)


car_data_points = DataSet()
car_data_points.loadData('./data/car.data2')

print car_data_points

data1 = car_data_points.numpyArrayData()
target1 = car_data_points.numpyArrayTarget()


from scipy import stats
def knnClassifier(training_data, test_data, training_target, test_target, k=5):
   #normalize the data
   #calculate the z-score of the data
   print training_data
   training_data = training_data
   new_training_data = stats.zscore(training_data.astype(int), axis=0)
   new_test_data = stats.zscore(test_data.astype(int), axis=0)
   #find the k nearest neighbors for each test data
   #print 'test', new_test_data
   predictions = []
   for test in new_test_data:
      #print test
      # find the euclidean distance between the test case and all training cases
      distances = []
      neighbors = []
      neighbor_predictions = []
      for train in new_training_data:
         #print train
         distances.append(np.linalg.norm(train-test))
      #print distances
      for i in range(k):
         neighb_i = distances.index(min(distances))
         neighbors.append(neighb_i)
         distances[neighb_i] = 1000000
      #print neighbors
      for neighb in neighbors:
         neighbor_predictions.append(training_target[neighb])
      predictions.append(stats.mode(neighbor_predictions)[0][0])
   return predictions
   
X_train, X_test, y_train, y_test = train_test_split(data1, target1, test_size=0.30, random_state=2)
predictions1 = knnClassifier(X_train, X_test, y_train, y_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
#print X_train, X_test, y_train, y_test
#print predictions
from sklearn.metrics import accuracy_score
print accuracy_score(predictions, y_test)
print accuracy_score(predictions1, y_test)