###...........................prediction of FoG  by using K-NN.....................................



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
t0= time.clock()


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier   #...knn classifier ......#

#.......... Import train_test_split function
from sklearn.model_selection import train_test_split       #...dividing data into training and testing data
from sklearn.preprocessing import StandardScaler 

#..........Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA         # importing PCA for dimensional reduction..
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA     # lda library

data = pd.read_csv("Book1.csv")                # loading csv file into data.
features= data.iloc[:, 1:10]                   #  extracting features from data.
target=data.iloc[:, 10]                        #  extracting target  from data.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.13)          # splitting data into 70% training and 30% test   

#...........scaling data...
sc_feature=StandardScaler()
X_train= sc_feature.fit_transform(X_train)    # train data after standardisation
X_test=sc_feature.transform(X_test)
 
t1=0
t2=0
#.............prediction using k-NN without PCA...............

def kNN_withoutPCA(traindata, testdata):
    print(".............k-NN without PCA...........")
    print(len(traindata))
    import math
    k=math.sqrt(len(y_test))
    print(k)
    
    #...Creating  KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=345, metric='euclidean')      #check for accuracy when no. of neighbour=5 or 7
    
    #..Training the model using the training sets
    knn.fit(traindata, y_train)
    
    #..Predict the response for test dataset
    print("X_test:", X_test)
    y_pred = knn.predict(testdata)
    print("Prediction with k-NN without pca:",y_pred)
   

    #..Evaluating  Model...
    cm=confusion_matrix(y_test, y_pred)
    print("Confusion of k-NN without PCA:\n",cm)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))     #### for measuring accuracy ###
    # print(f1_score(y_test, y_pred))



    t1 = time.clock() - t0       # time analysis
    print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)
    return y_pred, t1

#.................k-NN with PCA....................

def knn_PCA(traindata, testdata):
    print("............KNN with PCA................")
    
    # fit and transform data  dimension reduction
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(traindata)      # dimensional reduction with pca
    X_test_pca = pca.transform(testdata)
    print(len(X_train_pca))
    import math
    k=math.sqrt(len(y_test))
    print(k)
    
    #...Creating  KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=345, metric='euclidean')      #check for accuracy when no. of neighbour=5 or 7
    
    #..Training the model using the training sets
    knn.fit(X_train_pca, y_train)
    
    #..Predict the response for test dataset
    print("X_test_pca:", X_test_pca)
    y_pred_pca = knn.predict(X_test_pca)
    print("Prediction with k-NN with PCA:",y_pred_pca)
   

    #..Evaluating  Model...
    cm=confusion_matrix(y_test, y_pred_pca)
    print("Confusion of k-NN with PCA:\n",cm)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_pca))     #### for measuring accuracy ###
    # print(f1_score(y_test, y_pred))



    t2 = time.clock()-t0    # time analysis
    print("Time elapsed: ", t2) # CPU seconds elapsed (floating point)
    return y_pred_pca, t2


#.........................kNN with LDA.......................

def knn_LDA(traindata, testdata):
    print("............KNN with LDA................")
    
    # fit and transform data  dimension reduction
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(traindata, y_train)    #train data after lda reduction
    X_test_lda = lda.transform(testdata)



   
    print(len(X_train_lda))
    import math
    k=math.sqrt(len(y_test))
    print(k)
    
    #...Creating  KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=345, metric='euclidean')      #check for accuracy when no. of neighbour=5 or 7
    
    #..Training the model using the training sets
    knn.fit(X_train_lda, y_train)
    
    #..Predict the response for test dataset
    print("X_test_lda:", X_test_lda)
    y_pred_lda = knn.predict(X_test_lda)
    print("Prediction with k-NN with lda:",y_pred_lda)
   

    #..Evaluating  Model...
    cm=confusion_matrix(y_test, y_pred_lda)
    print("Confusion of k-NN with LDA:\n",cm)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_lda))     #### for measuring accuracy ###
    # print(f1_score(y_test, y_pred))



    t3= time.clock() - t0   # time analysis
    print("Time elapsed: ", t3) # CPU seconds elapsed (floating point)
    return y_pred_lda,t3



time1=0
time_pca=0
time_lda=0



y_knn, time1 =kNN_withoutPCA(X_train, X_test)    

y_knn_pca, time_pca=knn_PCA(X_train, X_test)

y_knn_lda, time_lda=knn_LDA(X_train, X_test)


print("t0:", t0)
print("time without dimension reduction:", time1)
print("time after pca:", time_pca-time1)
print("time after lda:", time_lda-time_pca)
# print("t2:", t2)

