# -*- coding: utf-8 -*-

import gestion_donnees as gd
from numpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from random import randint
from sklearn.model_selection import StratifiedShuffleSplit
import sys
from sklearn.metrics import accuracy_score


################################
# Execution en tant que script 
#
# taper python Perceptr.py
#
# dans un terminal
################################


class Perceptr:

    def __init__(self):
        #Plage de valeurs
        self.lamb = np.logspace(-5,0,9)
        self.learnRate = np.logspace(-5,0,9)


    #Entrainement du classifieur avec nos données d'entrainement
    def entrainement(self, X_train, y_train, clf):

        clf.fit(X_train, y_train)  # fit the model to the training data
        self.w = clf.coef_
        self.w_0 = clf.intercept_[0]
        #Insertion du biais
        self.w = np.insert(self.w, 0, self.w_0, axis=1)
        return clf

    #Prédiction d'une donnée
    def prediction(self, x, y):
        #Insertion du biais
        x = np.insert(x, 0, 1)
        #Prédiction
        result = np.dot((self.w),x.T)
        result = argmax(result) # valeur la plus élevée (score)
        return result

    #Prédiction d'un ensemble de données
    def predict_all(self, X_test, y_test):

        y_pred = np.array([self.prediction(x,y) for x,y in zip(X_test, y_test)]) #stocke les prédictions dans un tableau
        y_true = np.array([y for y in y_test]) #stocke les étiquettes dans un tableau
        accuracy_pred = accuracy_score(y_pred, y_true)
        return accuracy_pred

    def rechercheHypParm(self, X_train, y_train, X_test, y_test):
        #Récupération des divers paramètres
        lambdas = self.lamb
        learn_rates = self.learnRate
        historique = np.zeros((len(lambdas),len(learn_rates)))
        #Pour tous les paramètres
        i=0
        for lambd in lambdas :
            j=0
            for learn_rate in learn_rates:
                #créer Perceptron 
                perc = Perceptron(eta0=learn_rate, penalty='l2', alpha=lambd, random_state=0)
                #entrainement
                perc = self.entrainement(X_train, y_train, perc)
                #test
                historique[i][j] = self.predict_all(X_test, y_test) #affiche la précision pour chaque hyperparamètre
                j+=1
            i+=1
        print("Résultat des accuracy en fonction des différents paramètres lambda & learning rate :\n",historique)
        # On choisit les HP qui donne une précision maximale
        h = argmax(historique.reshape(len(lambdas)*len(learn_rates)))
        lambdindex=int(h/len(lambdas)) # Lamb correspondant à une précision max
        learnindex=h%(len(learn_rates)) # LR correspondant à une précision max

        print("Les meilleurs hyperparamètres choisis sont lambas = ",lambdas[lambdindex]," et learning rate = ", learn_rates[learnindex],"\n")
      
        # Affichage 
        self.affichage(historique, [lambdas[lambdindex], learn_rates[learnindex]]) 
        #Selection du perceptron avec les meilleurs hyper-parametres
        return Perceptron(eta0=learn_rates[learnindex], penalty='l2', alpha=lambdas[lambdindex], random_state=0)

    def cross_validation(self, train, labels, clf):
        #Création de plusieurs divisons de l'ensemble des donnees
        print("Validation croisée :")
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2) #n_spilt divisions, 0.2 = test 0.8 = train
        i=1
        meanScoreCV = 0
        for train_index, test_index in sss.split(train, labels): #pour chaque sous partie, on divise avec sss
            print("Itération ",i)
            i+=1
            X_train, X_test = train.values[train_index], train.values[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            #Entrainement + prediction pour chaque partie
            clf = self.entrainement(X_train, y_train, clf)
            score = self.predict_all(X_test, y_test)
            print("Précision : ",score)
            meanScoreCV += score
        #Score moyen
        meanScoreCV /= i-1
        print("La validation croisée affiche une prédiction moyenne de %.2f" %(meanScoreCV))

    #Affichage de la précision (taille des points) en fonction des hyperparamètres
    def affichage(self, historique, param = []):
        print("Les donnees sont affichees dans la figure.\nLe points rose determine la meilleure accuracy\n\n")
        for x in self.lamb: # affichage de toutes les combinaisons lamb - learnRate
            indexLamb = int(np.where(np.isclose(self.lamb,x))[0][0])
            tab = np.full(len(self.learnRate),x)
            plt.scatter(tab,self.learnRate,s=historique[indexLamb,:]*500, c ="black")
            
        plt.scatter(param[0],param[1], c = "pink")
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Précision en fonction des paramètres')
        plt.xlabel('Lambdas')
        plt.ylabel('Learning Rate')
        plt.show()


def main():
    #Création objet
    perceptron = Perceptr()
    #Récupération des donnees
    donnees = gd.GestionDonnees()
    [X_train, y_train, X_test, y_test, test, train, labels] = donnees.recupDonnees()
    #recherche des hyperparametres
    classifieur = perceptron.rechercheHypParm(X_train,y_train, X_test, y_test)
    #validation avec les HP trouvés précédemment
    perceptron.cross_validation(train, labels, classifieur)

if __name__ == "__main__":
    main()