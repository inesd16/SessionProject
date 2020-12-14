
"""  

################################
# Execution en tant que script 
#
# taper python SVM.py
#
# dans un terminal
################################

Chaimae Fillah
Ines Dobosz

"""
from sklearn.svm import SVC
from numpy import *
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import gestion_donnees as gd
from sklearn.model_selection import GridSearchCV
from random import randint
from sklearn.model_selection import train_test_split
from sklearn import decomposition

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


#Ce code a été pris sur le Kaggle, https://www.kaggle.com/udaysa/svm-with-scikit-learn-svm-with-parameter-tuning
#
# puis modifié pour adaptation à notre problème

class SVM : 
    def __init__(self):
      self.a = 0
      self.clf = None

    def hyperParameter(self,X_train, y_train):
        # Parametres par validation croisee
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                             'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                            {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                             'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                            {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                           ]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Réglage des hyper-parametres pour %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Meilleurs hyper parametres trouvés :")
            print()
            print(clf.best_params_)
            print()
            print("Grille des scores en fonction des HP :")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) pour %r"
                      % (mean, std * 2, params))
            print()
        #Renvoie le classifieurs avec les meilleurs HP
        self.clf = clf
        return clf

    #Prédiction des donnees de validation
    def prediction(self, X_train,y_train, X_test,y_test):
        #Entrainement
        sc= StandardScaler().fit(X_train)
        X_test_std = sc.transform(X_test)
        #Prediction
        y_test_pred = self.clf.predict(X_test_std)
        return y_test_pred

    def validationCroisee(self, train, labels):   
        #Création de plusieurs divisons de l'ensemble des donnees     
        print(" --- Validation croisee --- ")
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=21) #n_spilt divisions, 0.2 = test 0.8 = train
        i=1
        meanScoreCV = 0
        for train_index, test_index in sss.split(train, labels): #pour chaque sous partie, on divise avec sss
            print("Itération ",i)
            i += 1
            #print(" Index TRAIN:", train_index, "Index TEST:", test_index) #index des valeurs sélectionnées
            X_train, X_test = train.values[train_index], train.values[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            #Prediction pour chaque partie
            y_test_pred = self.prediction(X_train,y_train, X_test,y_test)

            #Récupération du score
            score = accuracy_score(y_test,y_test_pred)
            meanScoreCV += score
            print("Précision du modèle : ",accuracy_score(y_test,y_test_pred))

        #Score moyen pour toutes les divisions
        meanScoreCV /= i-1
        print("La validation croisée affiche une prédiction moyenne de %.2f" %(meanScoreCV))
        return meanScoreCV

    

def main():
    pca = decomposition.PCA()
    svm = SVM()

    #Récupération des donnees
    donnees = gd.GestionDonnees() 
    [X_train, y_train, X_test, y_test, test, train, labels] = donnees.recupDonnees()#stockage données csv dans variables

    #Selection du classifieurs svm avec les meilleurs parametres
    clf = svm.hyperParameter(X_train, y_train)

    #Validation
    svm.validationCroisee(train, labels)
    

if __name__ == "__main__":
    main()
