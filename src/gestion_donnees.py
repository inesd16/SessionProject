# -*- coding: utf-8 -*-

#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
###

import numpy as np


class GestionDonnees:
    def __init__(self):
        self.nb_train = 0
        self.nb_test = 0
        self.nb_split = 1
        self.test_size = 0.2
        self.train_add = '../leaf-classification/train.csv'
        self.test_add = '../leaf-classification/test.csv'


        if not self.lineairement_sep:
            x_2 = np.random.randn(nb_data_2, 2) + np.array([[0, 4]])  # Gaussienne centrée en mu_1_1=[0,4]
            t_2 = np.ones(x_2.shape[0])

            # Fusionne toutes les données dans un seul ensemble
            x = np.vstack([x, x_2])
            t = np.hstack([t, t_2])

        # Mélange aléatoire des données
        p = np.random.permutation(len(t))
        x = x[p, :]
        t = t[p]

        return x, t

    def generer_donnees(self):
        """
        Fonction qui genere des donnees de test et d'entrainement.
        nb_train : nb de donnees d'entrainement
        nb_test : nb de donnees de test
        """
        x_train, t_train = self.donnees_aleatoires(self.nb_train)
        x_test, t_test = self.donnees_aleatoires(self.nb_test)

        return x_train, t_train, x_test, t_test