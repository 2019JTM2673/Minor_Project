###....prediction of FoG  by using K-NN#############3



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
t0= time.clock()


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier   #...knn classifier ......#
# Import train_test_split function
from sklearn.model_selection import train_test_split       #...dividing data into training and testing data
from sklearn.preprocessing import StandardScaler 
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


data = pd.read_csv("Book1.csv")
# print(len(data))

# f_ax=data['ax']
# f_ay=data['ay']
# f_az=data['az']

# f_bx=data['bx']
# f_by=data['by']
# f_bz=data['bz']

# f_cx=data['cx']
# f_cy=data['cy']
# f_cz=data['cz']
# # features1 = pd.read_csv("Book1.csv", usecols=['Time','ax','ay','az','bx','by','bz','cx','cy','cz' ])   #training data
# # # # print(features)

# features=list(zip(f_ax, f_ay, f_az, f_bx, f_by, f_bz, f_cx, f_cy, f_cz,))     # combininig all features...
# print("feature=",features)


# target= list(data['state'])
# # print(target)
features= data.iloc[:, 1:10]
target=data.iloc[:, 10]
print("feature=",features)

# target=target.replace(0,np.NaN)
# mean= int(target.mean(skipna=True))
# target= target.replace(np.NaN, mean)




# # Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.13) # 70% training and 30% test

## scaling data...
sc_feature=StandardScaler()
X_train= sc_feature.fit_transform(X_train)
X_test=sc_feature.transform(X_test)

print(len(X_train))
import math
k=math.sqrt(len(y_test))
# print(k)


# # print(X_train)
# # print(y_train[0:5])
# # print(y_test[0:5])

# #Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=345, metric='euclidean')      #check for accuracy when no. of neighbour=5 or 7



# # #Train the model using the training sets
knn.fit(X_train, y_train)

# # #Predict the response for test dataset
y_pred = knn.predict(X_test)
print("prediction:",y_pred)
# # print(y_pred.shape)

#..Evaluate Model...
cm=confusion_matrix(y_test, y_pred)
print("confusion:",cm)



# # Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))     #### for measuring accuracy ###
# print(f1_score(y_test, y_pred))


t1 = time.clock() - t0
print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)
