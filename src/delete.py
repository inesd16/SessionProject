#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import StratifiedShuffleSplit
#import numpy as np
#from sklearn.svm import SVC
#from sklearn import svm
#import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    x = [5,6,7,8]
    y = [1,3,5,2]
    print("test")
    plt.figure(figsize=(30,30))
    plt.scatter(1,2)

    plt.title('Nuage de points avec Matplotlib')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('ScatterPlot_05.png')
    plt.show()