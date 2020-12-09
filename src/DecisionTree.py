from numpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from random import randint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

def recupDonnees():
    train = pd.read_csv(r'C:\Users\inesd\source\repos\SessionProject\leaf-classification\train.csv') #on récupère  les données d'entrainment (990)
    test = pd.read_csv(r'C:\Users\inesd\source\repos\SessionProject\leaf-classification\test.csv') #on récupère  les données d'entrainment (594)
    #print(train.describe())
    #print(test.describe())
    train, labels, test, test_ids, classes = encode(train, test)
    train.head(1)

    final_test = np.zeros((len(test),64*3))
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=23) #n_spilt divisions, 0.2 = test 0.8 = train
        
    for i in range(0,len(test)):
        final_test[i] = test.values[i]

    for train_index, test_index in sss.split(train, labels): #pour chaque sous partie, on divise avec sss
        #print(" Index TRAIN:", train_index, "Index TEST:", test_index) #index des valeurs sélectionnées
        X_train, X_test = train.values[train_index], train.values[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
    #print("TRAIN:", X_train, "TEST:", X_test, "taille x_train", X_train.shape, "taille x_test", X_test.shape)
    #print("TRAIN:", X_train, "TEST:", X_test, "taille x_train", len(X_train), "taille x_test", len(X_test))
    #print("TRAIN:", y_train, "TEST:", y_test, "taille y_train", y_train.shape, "taille y_test", y_test.shape)
    nb_train = len(X_train)
    nb_test = len(X_test)
    test = len(final_test)
    return (X_train, y_train, X_test, y_test, final_test)

# Organisation des donnees
def donnes():
    # Mélange dans un ordre aléatoire
    p = np.random.permutation(len(y_train))
    X_train = X_train[p, :]
    y_train = y_train[p]

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode les especes 
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    #print(classes)
    #print(train.species)
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    print("test", test)
    print("Longeur test", len(test))
    #print(train)
    #print(test)
    
    return train, labels, test, test_ids, classes

[X_train, y_train, X_test, y_test, test] = recupDonnees()#stockage données csv dans variables
# dividing X, y into train and test data 
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
  
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions)
for i in range(0,99):
    print(cm[:,i])