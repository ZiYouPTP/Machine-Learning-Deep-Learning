from sklearn import datasets, neighbors 
import numpy as np

import math
import operator

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

iris = datasets.load_iris()

def calculate_distance(p1,p2):
	distance = 0
	dimension = len(p1)
	for i in range(dimension):
		distance += (p1[i] - p2[i])*(p1[i] - p2[i])
	return math.sqrt(distance) 

def get_k_neighbors(training_x, label_y, point,k):
	distances = []
	neighbors = []
	for i in range(len(training_x)):
		distance = calculate_distance(training_x[i],point)
		distances.append(distance)

	index = []
	while len(neighbors) < k:
		i = 0
		min_distance = 999999
		min_idx = 0
		while i < len(distances):
			if i in index:
				i+=1
				continue
			if distances[i] < min_distance:
				min_distance = distances[i]
				min_idx = i 

			i+=1

		#add min index so we skip it in the next iteration
		index.append(min_idx)
		neighbors.append(label_y[min_idx])
	return neighbors

	distances.sort(key=operator.itemgetter(0))
	for i in range(k):
		neighbors.append(distances[i][1])
	return neighbors

def highest_votes(labels):
	if not labels:
		return None
	counts = {}
	for label in labels:
		counts[label] = counts.get(label,0) + 1 
	most_voted = max(counts, key = counts.get)
	return most_voted 

def predict(training_x, label_y, point, k):
	neighbors_labels = get_k_neighbors(training_x, label_y, point,k)
	return highest_votes(neighbors_labels)

def accuracy_score(predict, labels):
	total = len(predict)
	correct_count = 0
	for i in range(total):
		if predict[i] == labels[i]:
			correct_count +=1
	accuracy = correct_count/total
	return accuracy

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

randIndex = np.arange(iris_x.shape[0])
np.random.shuffle(randIndex)

iris_x = iris_x[randIndex]
iris_y = iris_y[randIndex]

x_train = iris_x[:100,:]
x_test = iris_x[100:,:]
y_train = iris_y[:100]
y_test = iris_y[100:]

k = 5
y_predict = []
for p in x_test:
	label = predict(x_train,y_train,p,k)
	y_predict.append(label)

print("accuracy of k: %.2f %%" %(100*accuracy_score(y_predict,y_test)))

# iris_X = iris.data
# iris_y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

# print ("Training size: %d" %len(y_train))
# print ("Test size    : %d" %len(y_test))

# def myweight(distances):
#     sigma2 = .5 # we can change this number
#     return np.exp(-distances**2/sigma2)

# clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print ("Accuracy of 10NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))