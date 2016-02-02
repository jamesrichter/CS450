from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy
class Node:
   def __init__(self):
      self.feature = None
      self.children = {}
   def __repr__(self):
      return self.feature

def printTree(root, number_of_tabs=0):
   leaves = []
   for node,child in root.children.items():
      for x in range(number_of_tabs):
          print '\t',
      print "feature:", root.feature
      try:
         for x in range(number_of_tabs):
            print '\t',
         print "choose:", node
         leaves = leaves + printTree(child, number_of_tabs+1)
      except:
         for x in range(number_of_tabs):
            print '\t',
         print "\tend value:", child
   return leaves

def getAllLeaves(root):
   leaves = []
   for _,child in root.children.items():
      try:
         leaves = leaves + getAllLeaves(child)
      except:
         leaves.append(child)
   return leaves

def traverseTree(datapoint, root):
   try:
      child = root.children[datapoint[root.feature]]
      newdata = np.delete(datapoint, root.feature)
      return traverseTree(newdata, child)
   except:
      try:
         return scipy.stats.mode(getAllLeaves(root))[0][0]
      except:
         return root

def calc_entropy(p):
   if p!=0:
      return -p * np.log2(p)
   else:
      return 0

def discretizeData(data):
   for feature in range(len(data[0])):
      column = data[:,feature]
      floatData = True
      for entry in column:
         try:
            float(entry)
         except:
            floatData = False
            break
      if floatData:
         column = column.astype(np.float32)
         moreThan10 = False
         uniqueValues = []
         for entry in column:
            if entry not in uniqueValues:
               uniqueValues.append(entry)
               if len(uniqueValues) > 10:
                  moreThan10 = True
                  break
         if moreThan10:
            s = np.std(column)
            m = np.mean(column)
            maxC = max(column)
            discreteArray = [m-2*s, m-1.5*s, m-s, m-.5*s, m, m+.5*s, m+s, m+1.5*s, m+2*s, maxC+1]
            for i in range(len(column)):
               for j in discreteArray:
                  if column[i] < j:
                     data[i,feature] = j
                     break

   return data


def make_tree(data, target):
   if np.shape(data)[1] == 1:
      return scipy.stats.mode(target)[0][0]
   best_entropy = [None, 100000]
   datalength = len(data)
   # find the best data feature
   for feature in range(len(data[0])):
      column = data[:,feature]
      entropy = 0
      # find out the feature's entropy
      uniques = np.unique(column)
      for choice in uniques:
         # locations where the column represents our choice
         # make comments on paper :3
         # so you can draw a chart
         locations = np.where(column == choice)         
         num_chosen = len(sum(locations))
         weight = 1 - num_chosen/float(datalength)
         outcomes = np.unique(target[locations])
         num_outcomes = len(outcomes)
         for outcome in outcomes:
            num_outcome = len(sum(np.where(target[locations]==outcome)))
            p = num_outcome / float(num_chosen)
            entropy += weight * calc_entropy(p)
      if entropy < best_entropy[1]:
         best_entropy = [feature , entropy, uniques]
   tree = Node()
   tree.feature = best_entropy[0]
   for unique in best_entropy[2]:
      locs = np.where(data[:,tree.feature]==unique)
      newdata = data[locs]
      newdata = scipy.delete(newdata, tree.feature, 1)
      newtarget = target[locs]
      tree.children[unique] = make_tree(newdata, newtarget)

   return tree

def homemadeDecisionTree(house_data, house_target, test_data, test_target):
   tree = make_tree(discretizeData(house_data), house_target)
   num_right = 0
   num_total = 0
   for i in range(test_data.__len__()):
      #print traverseTree(test_data[i], tree)
      if traverseTree(test_data[i], tree) == test_target[i]:
         num_right += 1
      num_total += 1
   print  "homemade accuracy:", num_right / float(num_total)

def accuracyClf(clf_target, test_target):
   num_right = 0
   num_total = 0
   for i in range(clf_target.__len__()):
      if clf_target[i] == test_target[i]:
         num_right += 1
      num_total += 1
   print "CLF accuracy:", num_right / float(num_total)
      

# IRIS classifier
iris = load_iris()

train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.30, random_state=2)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
accuracyClf(clf.predict(test_data[:, :]), test_target)
homemadeDecisionTree(train_data, train_target, test_data, test_target)


# HOUSE VOTES classifier
array = np.loadtxt('./data/house-votes-84.data', dtype = "U", delimiter=',')

array[array == "y"] = "1"
array[array == "n"] = "-1"
array[array == "?"] = "0"
house_data = array[:,1::]
house_target = array[:,0]
train_data, test_data, train_target, test_target = train_test_split(house_data, house_target, test_size=0.30, random_state=2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
accuracyClf(clf.predict(test_data[:, :]), test_target)
homemadeDecisionTree(train_data, train_target, test_data, test_target)





# CHESS MOVES classifier

array = np.loadtxt('./data/krkopt.data', dtype = "U", delimiter = ',')

array[array == 'a'] = '1'
array[array == 'b'] = '2'
array[array == 'c'] = '3'
array[array == 'd'] = '4'
array[array == 'e'] = '5'
array[array == 'f'] = '6'
array[array == 'g'] = '7'
array[array == 'h'] = '8'
house_data = array[:,0:-1]
house_target = array[:,-1]
train_data, test_data, train_target, test_target = train_test_split(house_data, house_target, test_size=0.30, random_state=2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
accuracyClf(clf.predict(test_data[:, :]), test_target)
homemadeDecisionTree(train_data, train_target, test_data, test_target)





# CREDIT CARD classifier

array = np.loadtxt('./data/crx.data', dtype = "U", delimiter = ',')

array[array == 'a'] = '1'
array[array == 'b'] = '2'
array[array == 'c'] = '3'
array[array == 'd'] = '4'
array[array == 'e'] = '5'
array[array == 'f'] = '6'
array[array == 'g'] = '7'
array[array == 'h'] = '8'
array[array == 'i'] = '9'
array[array == 'j'] = '10'
array[array == 'k'] = '11'
array[array == 'l'] = '12'
array[array == 'm'] = '13'
array[array == 'n'] = '14'
array[array == 'o'] = '15'
array[array == 'p'] = '16'
array[array == 'q'] = '17'
array[array == 'r'] = '18'
array[array == 's'] = '19'
array[array == 't'] = '20'
array[array == 'u'] = '21'
array[array == 'v'] = '22'
array[array == 'w'] = '23'
array[array == 'x'] = '24'
array[array == 'y'] = '25'
array[array == 'z'] = '26'
array[array == 'ff'] = '27'
array[array == 'bb'] = '28'
array[array == 'cc'] = '29'
array[array == 'aa'] = '30'
array[array == 'dd'] = '31'
array[array == 'gg'] = '31'
array[array == 'hh'] = '31'
array[array == '?'] = '-1'
house_data = array[:,0:-1]
house_target = array[:,-1]
train_data, test_data, train_target, test_target = train_test_split(house_data, house_target, test_size=0.30, random_state=2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
accuracyClf(clf.predict(test_data[:, :]), test_target)
homemadeDecisionTree(train_data, train_target, test_data, test_target)


# LENSES classifier

array = np.loadtxt('./data/lenses.data', dtype = "U")
house_data = array[:,0:-1]
house_target = array[:,-1]
train_data, test_data, train_target, test_target = train_test_split(house_data, house_target, test_size=0.30, random_state=2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
accuracyClf(clf.predict(test_data[:, :]), test_target)
homemadeDecisionTree(train_data, train_target, test_data, test_target)


pass