
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

"""

Changer self.train_add & self.test_add avec le chemin brut
test

"""

class GestionDonnees:
    def __init__(self):
        self.nb_train = 0
        self.nb_test = 0
        self.nb_split = 1
        self.test_size = 0.2
        self.train_add = r'C:\Users\inesd\source\repos\SessionProject\leaf-classification\train.csv'
        self.test_add = r'C:\Users\inesd\source\repos\SessionProject\leaf-classification\test.csv'


    def recupDonnees(self):
        #Récupération des donnees d'entrainement et de test à partir des fichiers csv
        train = pd.read_csv(self.train_add) #on récupère  les données d'entrainment (990)
        test = pd.read_csv(self.test_add) #on récupère  les données d'entrainment (594)
        train, labels, test, test_ids, classes = self.encode(train, test)
        train.head(1)

        final_test = np.zeros((len(test),64*3))
        sss = StratifiedShuffleSplit(n_splits=self.nb_split, test_size=self.test_size, random_state=1) #n_split divisions, 0.2 = test 0.8 = train
        
        #Valeur de test (sans étiquette de classe)
        for i in range(0,len(test)):
            final_test[i] = test.values[i]

        i=0
        #Mélange aléatoire
        for train_index, test_index in sss.split(train, labels): #pour chaque sous partie, on divise avec sss

            X_train, X_test = train.values[train_index], train.values[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        
        self.nb_train = len(X_train)
        self.nb_test = len(X_test)
        self.test = len(final_test)
        return (X_train, y_train, X_test, y_test, final_test, train, labels)
    
    def encode(self, train, test):
        le = LabelEncoder().fit(train.species) 
        labels = le.transform(train.species)           # encode les especes 
        classes = list(le.classes_)                    # save column names for submission
        test_ids = test.id                             # save test ids for submission
        train = train.drop(['species', 'id'], axis=1)  
        test = test.drop(['id'], axis=1)
    
        return train, labels, test, test_ids, classes