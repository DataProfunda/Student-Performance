import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from MultiClassifierModule import MultiClassifier


#Reading csv file
df = pd.read_csv('student-mat.csv')     
df2 = pd.read_csv('student-por.csv')
df3 = pd.concat([df,df2]) # Concatenating two dataset into one

data_prep = df3.copy()


#Full dictionary for mapping categorical data
d_full = {'school': {'GP': 0, 'MS': 1},
 'sex': {'F': 0, 'M': 1},
 'address': {'U': 0, 'R': 1},
 'famsize': {'GT3': 0, 'LE3': 1},
 'Pstatus': {'A': 0, 'T': 1},
 'Mjob': {'at_home': 0,  'services': 1,'teacher': 2, 'health': 3, 'other': 4},
 'Fjob': {'at_home': 0,  'services': 1,'teacher': 2, 'health': 3, 'other': 4},
 'reason': {'course': 0, 'other': 1, 'home': 2, 'reputation': 3},
 'guardian': {'mother': 0, 'father': 1, 'other': 2},
 'schoolsup': {'no': 0, 'yes': 1},
 'famsup': {'no': 0, 'yes': 1},
 'paid': {'no': 0, 'yes': 1},
 'activities': {'no': 0, 'yes': 1},
 'nursery': {'no': 0, 'yes': 1},
 'higher': {'no': 0, 'yes': 1},
 'internet': {'no': 0, 'yes': 1},
 'romantic': {'no': 0, 'yes': 1}}

for x in data_prep.columns:
    if(data_prep[x].dtype == np.object):    #If column has object type it has to be mapped
        d = d_full[x]                              #Dictionary for mapping one column
        data_prep[x] = data_prep[x].map(d)



scaler = MinMaxScaler() #With MinMaxScaler we scale values in each column into 0-1
scaler.fit(data_prep)
X_std= scaler.transform(data_prep)
X_std = pd.DataFrame(X_std, columns = data_prep.columns)

target = X_std['sex'].copy().astype(int)
train = X_std.drop('sex', axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.15)


multi_clf = MultiClassifier(X_train, X_test, y_train, y_test,n_repetition=10 )

multi_clf.compile_fit()

multi_clf.evaluate()
