
"""

################################
# Execution en tant que script 
#
# taper python NaiveBayes.py
#
# dans un terminal
################################

Chaimae Fillah
Ines Dobosz

"""

from numpy import *
import numpy as np
import gestion_donnees as gd
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score



class NaiveBayes:

    def __init__(self):
        #Plage de valeurs
        self.alpha = np.logspace(-5,0,20)

    def choixNB(self, X_train, y_train, X_test, y_test):
        accuracy_array = np.zeros((len(self.alpha)))
        i=0
        for param in self.alpha:
            classifieur = BernoulliNB(alpha = param)
            classifieur.fit(X_train,y_train)
            accuracy_array[i] = classifieur.score(X_test, y_test)
            i+=1

        print("\n\nTableau des scores pour chaque paramètre : ",accuracy_array, "\n")
        print("La meilleure précision est : ", max(accuracy_array), ",\nPour un paramètre alpha qui vaut : ",self.alpha[argmax(accuracy_array)], "\n")

        return BernoulliNB(alpha = self.alpha[argmax(accuracy_array)])

    
    #Validation croisee
    def validation_croisee(self, train,
                          labels, clf, cv):
        #Division en 10 sous parties
        scoreCV = cross_val_score(clf, train, labels, cv=cv)
        meanScoreCV = np.sum(scoreCV/len(scoreCV))
        print("La validation croisée affiche une prédiction moyenne de %.2f" %(meanScoreCV))




def main():

    #Creation de l'objet
    nb = NaiveBayes()
    #Récupération des données
    donnees = gd.GestionDonnees()
    [X_train, y_train, X_test, y_test, test, train, labels] = donnees.recupDonnees()

   # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    

    classifieur = nb.choixNB(X_train, y_train,
                            X_test, y_test)
    nb.validation_croisee(train, labels, 
                          classifieur, 10)

    
if __name__ == "__main__":
    main()


