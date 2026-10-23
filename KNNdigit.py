from sklearn import datasets, neighbors 
import numpy as np
import math
import operator
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

digit = datasets.load_digits()
digit_X = digit.data 
digit_y = digit.target 

X_train, X_test, y_train, y_test = train_test_split(digit_X, digit_y, test_size=len(digit_X)//5)

print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))

def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Accuracy of 10NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

plt.gray()
plt.imshow(X_test[0].reshape(8,8))
print(clf.predict(X_test[0].reshape(1,-1)))
plt.show()