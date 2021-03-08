import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from MultiClassifierModule import MultiClassifier

df = pd.read_csv('student-mat.csv')

df2 = pd.read_csv('student-por.csv')
df3 = pd.concat([df,df2]) # Concatenating two dataset into one

data_prep = df3.copy()

d_full = {} #Dictionary for mapping dictionaries
for x in data_prep.columns:
    if(data_prep[x].dtype == np.object):    #If column has object type it has to be mapped
        d = {}                              #Dictionary for mapping one column
        for y in range(len(pd.unique(data_prep[x]))):
            d[ pd.unique(data_prep[x])[y] ] = y
        data_prep[x] = data_prep[x].map(d)
        d_full[x] = d



scaler = MinMaxScaler() #With MinMaxScaler we scale values in each column into 0-1
scaler.fit(data_prep)
X_std= scaler.transform(data_prep)
X_std = pd.DataFrame(X_std, columns = data_prep.columns)

target = X_std['sex'].copy().astype(int)
train = X_std.drop('sex', axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.15)


print()



multi_clf = MultiClassifier(X_train, X_test, y_train, y_test,n_repetition=20 )


multi_clf.compile_fit()

multi_clf.evaluate()
