# -*- coding: utf-8 -*-

#####
# Chaimae Fillah
# Ines Dobosz
# Mathieu Lussier
###

import numpy as np
import matplotlib.pyplot as plt
import random  # imported for cross-validation

"""
    Hyperparameter after conducting Cross-Validation using 
        nb_train = 300
        nb_test = 200
    
    --> lin_sep = 1
    lineaire kernel     - (self.lamb: 1e-09)
    
    --> lin_sep = 0
    rbf kernel          - (self.lamb: 1.0, self.sigma_square: 2.0)
    polynomial kernel   - (self.lamb: 1.0, self.M: 6.0, self.c: 1.0)
    sigmoidal kernel    - (self.lamb: 1e-08, self.b: 0.01, self.d: 1e-05)
"""


class MAPnoyau:
    def __init__(self, lamb=1.0, sigma_square=2.0, b=0.01, c=1, d=1e-05, M=6, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.
        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None


    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).
        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.
        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        #AJOUTER CODE ICI
        if self.noyau == "lineaire":  # Noyau lineaire
            kernel = lambda x, x_p: np.dot(x.T, x_p)
        elif self.noyau == "rbf":  # Noyau rbf - (self.sigma_square)
            kernel = lambda x, x_p: np.exp(-np.dot((x - x_p).T, (x - x_p)) / (2 * self.sigma_square))
        elif self.noyau == "polynomial":  # Noyau polynomial - (self.M, self.c)
            kernel = lambda x, x_p: (np.dot(x.T, x_p) + self.c)**self.M
        elif self.noyau == "sigmoidal":  # Noyau sigmoidal -  (self.b, self.d)
            kernel = lambda x, x_p: np.tanh(self.b * np.dot(x.T, x_p) + self.d)
        else:
            print('Parametre "noyau" inconnu -- Noyau par defaut')
            kernel = lambda x, x_p: 0  # Noyau par defaut

        N, D = x_train.shape
        K = np.zeros((N, N))
        for n in range(N):
            for m in range(N):
                K[n, m] = kernel(x_train[n, :], x_train[m, :])

        I = np.identity(x_train.shape[0])  # NxN
        self.a = np.linalg.solve((K + self.lamb * I), I).dot(t_train)
        self.x_train = x_train

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.
        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).
        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        #AJOUTER CODE ICI
        if self.noyau == "lineaire":  # Noyau lineaire
            kernel = lambda x, x_p: np.dot(x.T, x_p)
        elif self.noyau == "rbf":  # Noyau rbf - (self.sigma_square)
            kernel = lambda x, x_p: np.exp(-np.dot((x - x_p).T, (x - x_p)) / (2 * self.sigma_square))
        elif self.noyau == "polynomial":  # Noyau polynomial - (self.M, self.c)
            kernel = lambda x, x_p: (np.dot(x.T, x_p) + self.c)**self.M
        elif self.noyau == "sigmoidal":  # Noyau sigmoidal -  (self.b, self.d)
            kernel = lambda x, x_p: np.tanh(self.b * np.dot(x.T, x_p) + self.d)
        else:
            kernel = lambda x, x_p: 0  # Noyau par defaut

        k_x = np.array([kernel(x, x_p) for x_p in self.x_train])
        return np.dot(k_x.T, self.a) > 0.5

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (t - prediction) ** 2

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.
        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        # AJOUTER CODE ICI
        ## GRID SEARCH ##
        kernels_parameters = {
            'lineaire': {},
            'rbf': {'sigma_square': np.append(np.geomspace(0.000000001, 1.0, num=10), np.linspace(1, 2, 5)).tolist()},
            'polynomial': {'M': [2, 3, 4, 5, 6],
                           'c': [0, 1, 2, 3, 4, 5]},
            'sigmoidal': {'b': np.geomspace(0.00001, 0.01, 4).tolist(),
                          'd': np.geomspace(0.00001, 0.01, 4).tolist()}
        }
        _lamb = np.append(np.geomspace(0.000000001, 1.0, num=10), np.linspace(1, 2, 5)).tolist()

        if self.noyau == "lineaire":  # Noyau lineaire
            parameters_grid = np.array([_lamb]).T
        elif self.noyau == "rbf":  # Noyau rbf - (self.sigma_square)
            parameters_grid = np.array(np.meshgrid(_lamb,
                                                   kernels_parameters[self.noyau]['sigma_square'])).T.reshape(-1, 2)
        elif self.noyau == "polynomial":  # Noyau polynomial - (self.M, self.c)
            parameters_grid = np.array(np.meshgrid(_lamb,
                                                   kernels_parameters[self.noyau]['M'],
                                                   kernels_parameters[self.noyau]['c'])).T.reshape(-1, 3)
        elif self.noyau == "sigmoidal":  # Noyau sigmoidal -  (self.b, self.d)
            parameters_grid = np.array(np.meshgrid(_lamb,
                                                   kernels_parameters[self.noyau]['b'],
                                                   kernels_parameters[self.noyau]['d'])).T.reshape(-1, 3)

        k = 10  # as instruct in the comments
        X_length = len(x_tab)
        if X_length < k:
            k = X_length

        index_X = np.arange(X_length)
        random.shuffle(index_X)  # shuffle the index of X
        index_split = np.array_split(index_X, k)

        cross_validation_dict = dict()
        print("Grid Search processing...")
        for _param in parameters_grid:
            # Si le noyau est lineaire, seul l'hyperparametre lambda va etre considere
            self.lamb = _param[0]
            if self.noyau == "rbf":  # Noyau rbf
                self.sigma_square = _param[1]
            elif self.noyau == "polynomial":  # Noyau polynomial
                self.M = _param[1]
                self.c = _param[2]
            elif self.noyau == "sigmoidal":  # Noyau sigmoidal
                self.b = _param[1]
                self.d = _param[2]

            _errors = list()
            for _k in range(k):
                x_valid = x_tab[index_split[_k]]
                y_valid = t_tab[index_split[_k]]
                x_train = x_tab[np.concatenate(index_split[:_k] + index_split[_k+1:])]
                y_train = t_tab[np.concatenate(index_split[:_k] + index_split[_k+1:])]

                self.entrainement(x_train, y_train)
                predictions_valid = np.array([self.prediction(x) for x in x_valid])
                errors_valid = np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(y_valid, predictions_valid)])
                _errors.append(np.mean(errors_valid))
            cross_validation_dict[tuple(_param)] = np.mean(_errors)

        best_hyperparameters = min(cross_validation_dict, key=cross_validation_dict.get)


        if self.noyau == "lineaire":
            print("Using lineaire kernel")
        elif self.noyau == "rbf":  # Noyau rbf
            self.sigma_square = best_hyperparameters[1]
            print("Using rbf kernel")
            print("sigma_square: ", str(self.sigma_square))
        elif self.noyau == "polynomial":  # Noyau polynomial
            self.M = best_hyperparameters[1]
            self.c = best_hyperparameters[2]
            print("Using polynomial kernel")
            print("M: ", str(self.M))
            print("c: ", str(self.c))
        elif self.noyau == "sigmoidal":  # Noyau sigmoidal
            self.b = best_hyperparameters[1]
            self.d = best_hyperparameters[2]
            print("Using sigmoidal kernel")
            print("b: ", str(self.b))
            print("d: ", str(self.d))

        self.lamb = best_hyperparameters[0]
        print("lambda: ", str(self.lamb))

        self.entrainement(x_train, y_train)  # last training as instruct in the comments

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()