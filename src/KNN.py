# -*- coding: utf-8 -*-
from numpy import *
import gestion_donnees as gd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from random import shuffle
import sys

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

"""  

################################
# Execution en tant que script 
#
# taper python KNN.py 1 0
#
# dans un terminal
################################
python KNN.py 1 0

"""

class KNN : 
    etiquette_classe = 99
    def __init__(self):
        """
        Algorithmes des K plus proches voisins, classification lineaire"""

        self.nb_test = 0
        self.nb_train = 0
        self.test = 0

        
    
    #Prédiction de la classe des k plus proches voisins selon
    def predict_test(self, X_train, y_train, X_test, y_test, k): #k plus proches voisins
        if k==0: k=1
        print("Méthode k plus proches voisins, avec k = ", k)
        distances = np.zeros(self.nb_train)
        i=0
        score=0
        for imTest in X_test: #Pour tous les test 
            index=0
            resultat = np.zeros(self.etiquette_classe)
            sorted_list = np.zeros(self.nb_train)
            for imTrain in X_train: #pour tous les train de chaque test
                diff = imTest-imTrain
                distances[index] = np.linalg.norm(diff[0:63]) + np.linalg.norm(diff[64:127]) + np.linalg.norm(diff[128:191])
                #distances[index] = np.linalg.norm(diff)
                index+=1
            distances=distances/3
            sorted_list = np.argsort(distances) #trier par index des distances les plus petites
            voisinsProches = sorted_list[:k] # k distances les plus petites
            for j in range(0,k):
                resultat[y_train[voisinsProches[j]]] += 1
            result = np.argmax(resultat)
            #print("Prédiction : ", result, ", étiquette : ", y_test[i])
            if result==y_test[i]:
                score+=1
            i+=1
        print("Réussite: {}%".format(score/i*100))

    #Prédiction des k plus proches voisins en comparant les trois features (Margin, Shape, Texture)
    def predictByFeaturesKN(self, X_train, y_train, X_test, k, y_test = []):

        predictionsTot = np.zeros(len(X_test))
        distances = np.zeros(X_train.shape[0])

        #Comparaison des distances entre les donnees de test et d'entrainement
        i=0
        score=0
        for imTest in X_test:
            index=0
            resultat = np.zeros(self.etiquette_classe)
            sorted_list = np.zeros(self.nb_train)
            for imTrain in X_train:
                diff = imTest-imTrain
                # moyenne de chaque feature
                distances[index] = np.linalg.norm(diff[0:63]) + np.linalg.norm(diff[64:127]) + np.linalg.norm(diff[128:191])
                index+=1
            distances=distances/3
            sorted_list = np.argsort(distances) #trié par index des distances les plus petites
            voisinsProches = sorted_list #  distance la plus petite
            j=0

            #Tant que l'on n'a pas k voisins appartenant à la même classe
            while np.max(resultat)<k :
                resultat[y_train[sorted_list[j]]] += 1
                j+=1
            result = np.argmax(resultat)
            predictionsTot[i] = result

            #Prédiction pour des donnees sansp cible (donnees de test)
            if (y_test == []):
                print("Prédiction : ", result)
                i+=1

            #Prediction pour des donnees avec cible
            else :
                if result==y_test[i]:
                    score+=1
                i+=1
            #Précision totale
            scoreTot = score/i*100
        print("Réussite: {}%\n".format(score/i*100))
    
    
        return predictionsTot, scoreTot

    #Affichage des groupes de points
    #La couleur des points dépend de leur classe
    #Chaque dimension correspond à la moyenne d'une feature
    def affichage(self, X_train, y_train, test, k=1, y_test=[]):
        colors = []
        #On récupere les predictions
        predictionTest,_ = self.predictByFeaturesKN(X_train, y_train, test, 1, y_test)
        plt.figure(figsize=(30,30))
        #Affichage des points en 3D
        plt.axes(projection="3d")

        #Pour toutes les étiquettes de classe, on assigne une couleur différente
        for i in range(self.etiquette_classe):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        #tableau contenant donnees d'entrainement et donnees testees
        points = np.zeros((self.nb_train+self.test,3))
        i=0
        #Ajout des points d'entrainement
        for x in X_train:       
            points[i] = self.moyenneFeatures(x)
            c = colors[y_train[i]]      #différente couleur pour chaque etiquette de classe
            plt.scatter(points[i, 0], points[i, 1], points[i, 2], c=c) 
            i = i+1
        i=0

        #Ajout des donnees testees et ciblees sous forme de marqueur _
        for t in test:
            points[i+len(X_train)] = self.moyenneFeatures(t)
            c = colors[int(predictionTest[i])]      #différente couleur pour chaque etiquette de classe
            plt.scatter(points[i, 0], points[i, 1], points[i, 2], c=c, marker = '_')
            i = i+1

        plt.title('Margin Shape Texture')
        plt.show()
        

    #Calcul des moyennes de features pour une donnee
    def moyenneFeatures(self, x):
        moyMargin = np.linalg.norm(x[:64])
        moyShape = np.linalg.norm(x[64:128])
        moyTexture = np.linalg.norm(x[128:192])
        data = np.array([moyMargin,moyShape,moyTexture])
        return data

    def validation_croisee(self, train, labels):
        print(" --- Validation croisee --- ")
        #Création de plusieurs divisons de l'ensemble des donnees
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=23) #n_spilt divisions, 0.2 = test 0.8 = train
        i=0
        meanScoreCV = 0
        for train_index, test_index in sss.split(train, labels): #pour chaque sous partie, on divise avec sss
            i+=1
            print("Itération ",i)
            X_train, X_test = train.values[train_index], train.values[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            #Prediction pour chaque partie
            _,score = self.predictByFeaturesKN(X_train, y_train, X_test, 1, y_test)
            meanScoreCV += score
        meanScoreCV = meanScoreCV / (i)
        print("La validation croisée affiche une prédiction moyenne de %.2f \n" %(meanScoreCV))

    #Prédiction d'un ensemble pour tous les k
    def prediction_all_k(self, X_train, y_train, X_test, y_test):
        

        print(" -- Prédiction pour tous les k ---")
        print("Méthode des k plus proches voisins appartenant à la même classe")
        print("La classe prédite est celle qui correspond à la classe qu'on retrouve k fois, parmi les plus proches voisins")
        result = np.zeros(8)
        for k in range(1,9):
            print("Prédiction pour k = ",k)
            _,result[k-1] = self.predictByFeaturesKN(X_train, y_train, X_test, k, y_test)
        plt.scatter(range(1,9),result)
        plt.title('Précision en fonction du nombre de plus proches voisins désirés')
        plt.xlabel("Nombre de plus proches voisins")
        plt.ylabel("Précision")
        plt.show()




def main():
    
     #Récupération des parametres 
    if len(sys.argv) < 2:
        usage = "\n k nombre de voisins pour la prédiction\
        \n\t lin : 0: prédiction avec données de validation, 1: prédiction avec données de test"
        print(usage)
        return
    k = sys.argv[1]

    #Creation de l'objet
    knn = KNN()
    #Récupération des donnees
    donnees = gd.GestionDonnees() 
    [X_train, y_train, X_test, y_test, test, train, labels] = donnees.recupDonnees()#stockage données csv dans variables
    
    #Taille des échantillons
    knn.nb_train = len(X_train)
    knn.nb_test = len(X_test)
    knn.test = len(test)

    #Validation croisee
    knn.validation_croisee(train, labels)

    #Prédiction pour tous les k
    knn.prediction_all_k(X_train, y_train, X_test, y_test)

    
    #Donnees de test, sans cible
    if sys.argv[2] == 1:
        print("-- Prédicitions des données de test avec affichage --")
        knn.affichage(train, labels, test, k)

    #Donnees de validation
    else:
        print("-- Prédicitions des données de validation avec affichage --")
        knn.affichage(X_train, y_train, X_test, k, y_test)


if __name__ == "__main__":
    main()
