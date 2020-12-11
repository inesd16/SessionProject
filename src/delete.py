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

a=23
print(int(a/4)) #5
print(a%4) #3