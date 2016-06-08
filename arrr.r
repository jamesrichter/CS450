#Array indices start at 1 instead of 0
k = matrix(c(.0001, .001, .01, .05, .1, .5, 1, 10, 100, 1000))
C = matrix(c(.0001, .001, .01, .1, 1, 5, 10, 50, 100, 1000, 10000))

library(e1071)
letters <- read.csv("C:/Users/jim/Downloads/letters.csv", head=TRUE, sep=",")
allRows = 1:nrow(letters)
testRows = sample(allRows, trunc(length(allRows) * 0.3))
letters_test = letters[testRows,]
letters_train = letters[-testRows,]
best_k = 0
best_C = 0
best_accuracy = 0
for(i in k){
  for (j in C){
model = svm(letter~., data = letters_train, kernel="radial", gamma = i, cost = j, type="C")
prediction = predict(model, letters_test[,-1])
confusion_matrix = table(pred = prediction, true = letters_test$letter)
agreement = prediction == letters_test$letter
accuracy = prop.table(table(agreement))
if(best_accuracy < accuracy[2]){
  best_k = i
  best_C = j
  best_accuracy = accuracy[2]}
  }
  print(i)
}
model = svm(letter~., data = letters_train, kernel="radial", gamma = best_k, cost = best_C, type="C")
prediction = predict(model, letters_test[,-1])
confusion_matrix = table(pred = prediction, true = letters_test$letter)
print(confusion_matrix)
print(best_accuracy)
print(best_k)
print(best_C)

abalones <- read.csv("C:/Users/jim/Downloads/abalone.csv", head=TRUE, sep=",")
allRows = 1:nrow(abalones)
testRows = sample(allRows, trunc(length(allRows) * 0.3))
abalones_test = abalones[testRows,]
abalones_train = abalones[-testRows,]
best_k = 0
best_C = 0
best_accuracy = 0
for(i in k){
  for (j in C){
model = svm(Rings~., data = abalones_train, kernel="sigmoid", gamma = i, cost = j, type="C")
prediction = predict(model, abalones_test[,-9])
confusion_matrix = table(pred = prediction, true = abalones_test$Rings)
agreement = prediction == abalones_test$Rings
accuracy = prop.table(table(agreement))
if(best_accuracy < accuracy[2]){
  best_k = i
  best_C = j
  best_accuracy = accuracy[2]}
  }}
model = svm(Rings~., data = abalones_train, kernel="radial", gamma = best_k, cost = best_C, type="C")
prediction = predict(model, abalones_test[,-9])
confusion_matrix = table(pred = prediction, true = abalones_test$Rings)
print(confusion_matrix)
print(best_accuracy)
print(best_k)
print(best_C)

vowels <- read.csv("C:/Users/jim/Downloads/vowel.csv", head=TRUE, sep=",")
allRows = 1:nrow(vowels)
testRows = sample(allRows, trunc(length(allRows) * 0.3))
vowels_test = vowels[testRows,]
vowels_train = vowels[-testRows,]
best_k = 0
best_C = 0
best_accuracy = 0
for(i in k){
  for (j in C){
model = svm(Class~., data = vowels_train, kernel="radial", gamma = i, cost = j, type="C")
prediction = predict(model, vowels_test[,-13])
confusion_matrix = table(pred = prediction, true = vowels_test$Class)
agreement = prediction == vowels_test$Class
accuracy = prop.table(table(agreement))
if(best_accuracy < accuracy[2]){
  best_k = i
  best_C = j
  best_accuracy = accuracy[2]}
}}
model = svm(Class~., data = vowels_train, kernel="radial", gamma = best_k, cost = best_C, type="C")
prediction = predict(model, vowels_test[,-13])
confusion_matrix = table(pred = prediction, true = vowels_test$Class)
print(confusion_matrix)
print(best_accuracy)
print(best_k)
print(best_C)