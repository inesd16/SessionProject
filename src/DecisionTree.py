from numpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.metrics import confusion_matrix,accuracy_score
from random import randint
import gestion_donnees as gd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score


class DecisionTree:
    
    def __init__(self):
        """
        Algorithmes des K plus proches voisins, classification lineaire"""

        #Paramètres du classifieur
        self.criterions = ['gini', 'entropy']
        self.splitters = ['random','best']
        #Exécution des prédictions sur une place de max_depth (allant de depth_begin à depth_size)
        self.depth_begin = 1
        self.depth_size = 20
    

    #Entrainement du classifieur en fonction des caractéristiques
    def entrainement(self, 
                     X_train, y_train, depth = None, 
                     splitter = 'best', criterion = 'gini', clf = None):
        #Déclaration de l'arbre de décision
        clf = DecisionTreeClassifier(max_depth = depth, splitter = splitter,criterion=criterion).fit(X_train, y_train) 
        #Entrainement de l'abre de décision 
        clf.fit(X_train, y_train)
        return clf
        

    #Prédiction avec le classifieur particulier
    def prediction(self, X_test, clf):
        
        predictions = clf.predict(X_test) 
        return predictions


    def recherche_param(self, X_train, 
                        y_train, X_test, 
                        y_test):

        #Pour toutes les caractéristiques (paramètres)
        historique = np.zeros((2,2))
        #Tableau contenant les accuracy de chaque depth et les paramètres criterion et splitter correspondants
        bestParamDepth = np.zeros((self.depth_size-self.depth_begin,3))
        #Tableau contenant toutes les combinaisons de parametres
        historiqueTot = np.zeros(((self.depth_size-self.depth_begin)*4,3))
        #Regarder les précisions
        d=0
        for depth in range(self.depth_begin,self.depth_size):
            print("\nMax depth = ", depth)
            i=0
            for criterion in self.criterions:
                j=0
                for splitter in self.splitters:
                    clf = self.entrainement(X_train, y_train, 
                                    depth, splitter, criterion)
                    y_predicted = self.prediction(X_test,clf)
                    historique[i][j] = accuracy_score(y_predicted, y_test)
                    if (i==1 and j==0):
                        historiqueTot[d*4+3] = [historique[i][j],i,j]
                    else:
                        historiqueTot[d*4+i+j] = [historique[i][j],i,j]
                    j+=1
                i+=1
            #On retient les paramètres les plus efficaces, pour un même depth
            h = argmax(historique.reshape(2*2))
            crit = int(h/2) # criterion correspondant à une précision max
            split = h%(2) # splittter correspondant à une précision max

            #Stockage de la précision en fonction des paramètres
            bestParamDepth[d] = np.array([max(historique.reshape(2*2)),crit,split])
            d = d+1

        #Selection des paramètres ayant la meilleure précision parmi tous les depth
        acc_index = argmax(bestParamDepth[:,0]) #indice du meilleur accuracy
        #Triplet correspondant à la meilleur accuracy (acc, crit, splitter)
        bestParamDepth2 = bestParamDepth[acc_index]
        #Transformation des index en string (pour criterion et splitter)
        c,s = self.selectionParam(bestParamDepth2[1], bestParamDepth2[2])
        print("La meilleure précision ", bestParamDepth[acc_index, 0], 
              "est trouvée avec une depth = ",acc_index+self.depth_begin, 
              ",\nune criterion = ",c,
              ",\net un splitter = ",s)
        #On retourne le classifieur avec les meilleurs parametres
        clf = DecisionTreeClassifier(max_depth = acc_index+self.depth_begin, splitter = s,criterion=c)
        
        return clf, historiqueTot

    def selectionParam(self,index_criterion, index_splitter):
        return self.criterions[int(index_criterion)],self.splitters[int(index_splitter)]

        
    #Validation croisee
    def validation_croisee(self, train,
                          labels, clf):
        #Division en 10 sous parties
        scoreCV = cross_val_score(clf, train, labels, cv=10)
        meanScoreCV = np.sum(scoreCV/len(scoreCV))
        print("La validation croisée affiche une prédiction moyenne de %.2f" %(meanScoreCV))

    #Affichage des précisions en fonction des différents paramètres
    def affichageParam(self,historiqueTot):
        #Définir une couleur pour chaque jeu de paramètres
        color = ["cyan", "black", "magenta", "lightgreen"]
        a=0
        for p in historiqueTot:
            x=int(a/4)
            if (p[1]==1):
                if (p[2] == 1):
                    plt.scatter(x, p[0], c=color[0], label = 'criterion : entropy ; splitter : best') 
                else:
                    plt.scatter(x, p[0], c=color[1], label = 'criterion : entropy ; splitter : random') 
            if (p[1]==0):
                if (p[2] == 1):
                    plt.scatter(x, p[0], c=color[2], label = 'criterion : gini ; splitter : best') 
                else:
                    plt.scatter(x, p[0], c=color[3], label = 'criterion : entropy ; splitter : random') 
            a+=1
        print("La couleur bleue correspond à criterion : entropy ; splitter : best\
                \nLa couleur noire correspond à criterion : entropy ; splitter : random\
                \nLa couleur rose correspond à criterion : gini ; splitter : best\
                \nLa couleur verte correspond à criterion : entropy ; splitter : random\
                \n")
        plt.xlabel("max_depth")
        plt.ylabel("Précision")
        plt.title("Accuracy en fonction des paramètres")
        plt.show()

def main():
    #Récupération des donnees
    donnees = gd.GestionDonnees()
    [X_train, y_train, X_test, y_test, test, train, labels] = donnees.recupDonnees()#stockage données csv dans variables
    #Création d'un objet
    tree = DecisionTree()
    #Recherche du meilleur classifieur
    clf,afficher = tree.recherche_param(X_train, 
                    y_train, X_test,
                    y_test)

    #Affichage des test pour les paramètresp
    tree.affichageParam(afficher)

    #Valisation croisee
    tree.validation_croisee(train, labels, clf)

    #Prédiction du des donnees de test
    clf.fit(X_train,y_train)
    pred = tree.prediction(test, clf)
    print("Prédictions faites sur les données de test : ",pred)
        
if __name__ == "__main__":
    main()
