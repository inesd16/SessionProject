# -*- coding: utf-8 -*-

""" 
################################
# Execution en tant que script 
#
# taper python Combinaison.py
#
# dans un terminal
################################

Chaimae Fillah
Ines Dobosz 

"""

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from random import randint
import gestion_donnees as gd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import DecisionTree as dt
import NaiveBayes as nb
import Perceptr as perceptr
import KNN as knn
import SVM as svm



class Combinaison : 
    def __init__(self):


        """        Algorithmes combinant différents modèles (5 au total), classification lineaire       """

        self.etiquette_classe = 99

    def choix_classifieurs(self, X_train, y_train, X_test, y_test):

        print(" \n\t\t--- Recherche des meilleurs classifieurs pour chaque méthode ---\n\n")

        #Choix des classifieurs

        print(" --- Recherche pour Naive Bayes ---\n")
        #Naive Bayes
        nB = nb.NaiveBayes()
        clfNB = nB.choixNB(X_train, y_train, X_test, y_test)
        
        
        #Arbre de décision
        print(" --- Recherche pour Arbre de Decision ---\n")
        tree = dt.DecisionTree()
        clfTree,_ = tree.recherche_param(X_train, y_train, 
                                        X_test, y_test)


        #K plus proches voisins
        print("\n --- Pas de recherche de paramètres pour les K plus proches voisins ---\n")
        kNN = knn.KNN()


        #SVM
        print(" --- Recherche pour la SVM ---\n")
        sVM = svm.SVM()
        clfSVM = sVM.hyperParameter(X_train, y_train)


        #Perceptron
        print(" --- Recherche pour le Perceptron ---\n")
        perceptron = perceptr.Perceptr()
        clfPerceptr = perceptron.rechercheHypParm(X_train, y_train, X_test, y_test)

        return (clfNB, clfTree, kNN, clfPerceptr, clfSVM)

    
    def predictionsTest(self, X_train, y_train, X_test, y_test, test, train, labels):

        print("--Prédictions Test--")
        #Entrainement avec l'entiereté des données d'entrainement
        #Test sur les donnée de test
        #On récupère les différents classifieurs optimisés par leurs hyperparamètres
        clfNB, clfTree, kNN, clfPerceptr, clfSVM = self.clfNB, self.clfTree, self.kNN, self.clfPerceptr, self.clfSVM

        #On les entraîne
        clfNB.fit(train,labels)
        clfTree.fit(train,labels)

        clfPerceptr.fit(train,labels)
        clfSVM.fit(train,labels)

        #Prédiction pour chaque classifieur, pour les données de test
        predictionNB = clfNB.predict(test)
        predictionTree = clfTree.predict(test)
        predictionKNN,_ = kNN.predictByFeaturesKN(X_train, y_train, test, 1)
        predictionPerceptr = clfPerceptr.predict(test)
        predictionSVM = clfSVM.predict(test)

        #Comparaison de tous les résultats
        #On prédit la classe qui revient le plus pour chaque donnée
        etiquetteFinale = np.zeros(len(predictionNB))
        for i in range(0,len(predictionNB)):
            #contient le nombre d'occurrence de chaque classe pour l'entree i
            occurence_class = np.zeros(self.etiquette_classe)

            #La classe prédite pour chaque classifieur est ajoutée au tableau
            occurence_class[predictionNB[i]] += 1
            occurence_class[predictionTree[i]] += 1
            occurence_class[int(predictionKNN[i])] += 1
            occurence_class[predictionPerceptr[i]] += 1
            occurence_class[predictionSVM[i]] += 1

            #La classe la plus redondante est prédite
            etiquetteFinale[i] = argmax(occurence_class)
        print(etiquetteFinale)


    def predictionsValidation(self, X_train, y_train, X_test, y_test):
        
        print("--Prédiction Validation--")

        #On récupère les différents classifieurs optimisés par leurs hyperparamètres
        clfNB, clfTree, kNN, clfPerceptr, clfSVM = self.clfNB, self.clfTree, self.kNN, self.clfPerceptr, self.clfSVM

        #On les entraîne
        clfNB.fit(X_train, y_train)
        clfTree.fit(X_train, y_train)

        clfPerceptr.fit(X_train, y_train)
        clfSVM.fit(X_train, y_train)

        #Prédiction pour chaque classifieur, pour les données de validation
        
        print("\n # Prédiction Naive Bayes # \n")
        predictionNB = clfNB.predict(X_test)
        print(" # Prédiction Arbre de décision # \n")
        predictionTree = clfTree.predict(X_test)
        print(" # Prédiction KNN # \n")
        predictionKNN,_ = kNN.predictByFeaturesKN(X_train, y_train, X_test, 1, y_test)
        print("\n # Prédiction Perceptron # \n")
        predictionPerceptr = clfPerceptr.predict(X_test)
        print("\n # Prédiction SVM # \n")
        predictionSVM = clfSVM.predict(X_test)
        print("Voici les prédiction de chaque modèle, dans l'ordre, NaiveBayes, Arbre de Decision, K plus proches voisins, Perceptron, SVM : \n")
        print(predictionNB)
        print(predictionTree)
        print(predictionKNN)
        print(predictionPerceptr)
        print(predictionSVM)

        #Comparaison de tous les résultats
        #On prédit la classe qui revient le plus pour chaque donnée
        etiquetteFinale = np.zeros(len(predictionNB))
        for i in range(0,len(predictionNB)):
            #contient le nombre d'occurrence de chaque classe pour l'entree i
            occurence_class = np.zeros(self.etiquette_classe)

            #La classe prédite pour chaque classifieur est ajoutée au tableau
            occurence_class[predictionNB[i]] += 1
            occurence_class[predictionTree[i]] += 1
            occurence_class[int(predictionKNN[i])] += 1
            occurence_class[predictionPerceptr[i]] += 1
            occurence_class[predictionSVM[i]] += 1

            #La classe la plus redondante est prédite
            etiquetteFinale[i] = argmax(occurence_class)

        acc = accuracy_score(etiquetteFinale, y_test)
        print(etiquetteFinale)
        print("\n\nAccuracy finale de combinaison des modeles : ",acc*100,"%")


def main():
       
    #Création de l'objet
    combinaison_modeles = Combinaison()

    #Récupération des donnees
    donnees = gd.GestionDonnees() 
    [X_train, y_train, X_test, y_test, test, train, labels] = donnees.recupDonnees()#stockage données csv dans variables

    #Récupération des meilleurs classifieurs pour chaque méthode
    combinaison_modeles.clfNB, combinaison_modeles.clfTree, combinaison_modeles.kNN, combinaison_modeles.clfPerceptr, combinaison_modeles.clfSVM = combinaison_modeles.choix_classifieurs(X_train, y_train, X_test, y_test)

    #Prédiction
    pred_test = combinaison_modeles.predictionsTest(X_train, y_train, X_test, y_test, test, train, labels)
    pred_validation = combinaison_modeles.predictionsValidation(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
