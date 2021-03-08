#Multi clf + grid search cv


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sklearn


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow import keras




from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

import time

from sklearn.decomposition import IncrementalPCA

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import VotingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
import xgboost


from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import pickle


class MultiClassifier():
    
    def __init__(self,X_train,X_test, y_train, y_test, n_repetition=20, estimators = ('extra_clf', 'rnd_clf','voting_clf'), data_to_predict=None, col_id = None):
       
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_repetition = n_repetition
        self.estimators = estimators
        self.data_to_predict = data_to_predict
        self.col_id = col_id
        
    
    
    def compile_fit(self):
        
        self.clf_models = [] #List of classifiers, it is used for further fitting VotingClassifer
        

        if 'extra_clf' in self.estimators:       
            self.extra_compile_fit()
   
        if 'rnd_clf' in self.estimators:   
            self.rnd_compile_fit()    
            
            
        if 'voting_clf' in self.estimators:
            self.voting_compile_fit()
            
        

                
    def evaluate(self):  
        
        
        if 'extra_clf' in self.estimators:
            y_pred = self.extra_clf.predict(self.X_test)
            print("Extra Trees Clf", accuracy_score(self.y_test, y_pred) )
            
        if 'rnd_clf' in self.estimators:
            y_pred = self.rnd_clf.predict(self.X_test)
            print("Random Forest Clf", accuracy_score(self.y_test, y_pred))

        
        
        if 'voting_clf' in self.estimators:
            y_pred = self.voting_clf.predict(self.X_test)
            print("Voting Clf", accuracy_score(self.y_test, y_pred))

        
    def save_model(self):
        
        if 'extra_clf' in self.estimators:
            
            print("Do you want to save Extra Clf? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'extra_clf.sav'
                pickle.dump(self.extra_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
            
        if 'rnd_clf' in self.estimators:
            
            print("Do you want to save Random Clf? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'rnd_clf.sav'
                pickle.dump(self.rnd_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
     
        
        if 'voting_clf' in self.estimators:
            
            print("Do you want to save Voting Clf? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'voting_clf.sav'
                pickle.dump(self.voting_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
   

    
    def fit_with_test_data(self):
        
        if 'extra_clf' in self.estimators:
            self.extra_clf = pickle.load(open('extra_clf.sav', 'rb'))
            
        if 'rnd_clf' in self.estimators:       
            self.rnd_clf = pickle.load(open('rnd_clf.sav', 'rb'))
          
            

        if 'voting_clf' in self.estimators:
            
            self.rnd_clf.fit(self.X_test,self.y_test)
            self.extra_clf.fit(self.X_test,self.y_test)
            self.gb_clf.fit(self.X_test,self.y_test)
            
            self.voting_clf.fit(self.X_test,self.y_test)
            y_pred = self.voting_clf.predict(self.X_train)
            print("Voting Clf", accuracy_score(self.y_train, y_pred) / 1000000)
            

        
        print("dd")

    def load_models(self):
        
        if 'extra_clf' in self.estimators:
            self.extra_clf = pickle.load(open('extra_clf.sav', 'rb'))
            
        if 'rnd_clf' in self.estimators:       
            self.rnd_clf = pickle.load(open('rnd_clf.sav', 'rb'))
           

        if 'voting_clf' in self.estimators:
            
            self.gb_clf = pickle.load(open('voting_clf.sav', 'rb'))
            
            self.rnd_clf.fit(self.X_test,self.y_test)
            self.extra_clf.fit(self.X_test,self.y_test)
            self.gb_clf.fit(self.X_test,self.y_test)
            
            self.voting_clf.fit(self.X_test,self.y_test)
            y_pred = self.voting_clf.predict(self.X_train)
            print("Voting Clf", accuracy_score(self.y_train, y_pred) / 1000000)

    def predict_save(self, data_to_predict, col_id, predictor='voting_clf'):
        
        if predictor == 'extra_clf':
            
            submisson = pd.DataFrame( None, columns = ['Id','SalePrice'] )
            submisson['Id'] = col_id
            submisson['SalePrice'] = np.arange(len(data_to_predict))

            submisson['SalePrice'] = self.extra_clf.predict(data_to_predict).astype(int) 

            submisson.to_csv('submission.csv',index=False)

            print("Prediction saved!")
        
        if predictor ==  'rnd_clf':
            
            submisson = pd.DataFrame( None, columns = ['Id','SalePrice'] )
            submisson['Id'] = col_id
            submisson['SalePrice'] = np.arange(len(data_to_predict))

            submisson['SalePrice'] = self.extra_clf.predict(data_to_predict).astype(int) 

            submisson.to_csv('submission.csv',index=False)

            print("Prediction saved!")
            
        
        if predictor ==  'voting_clf':
            
            
            submisson = pd.DataFrame( None, columns = ['Id','SalePrice'] )
            submisson['Id'] = np.arange(len(self.data_to_predict))
            submisson['SalePrice'] = np.arange(len(self.data_to_predict))

            submisson['SalePrice'] = self.voting_clf.predict(self.data_to_predict).astype(int) 

            submisson['Id'] = self.col_id

            submisson.to_csv('submission.csv',index=False)

            print("Done!")

        
        
            
            
    def extra_compile_fit(self):
        param_grid = [ {"n_estimators" : [20,50, 70, 100,150,170,200,220],"max_depth":[ 2,3,10,15, 30, 40, 50 ] }]
            
        grid_search = GridSearchCV(ExtraTreesClassifier(), param_grid,  scoring="neg_mean_squared_error", verbose=2)
        grid_search.fit(self.X_train, self.y_train)
                
        print(grid_search.best_params_)
                
        y_pred = grid_search.predict(self.X_test)  
        mse = accuracy_score(self.y_test, y_pred)
        print(accuracy_score)
            
        self.extra_clf = 0
                
        prev_mse = 0     
        i = 0
        while(i < self.n_repetition):
                    
            if i == 0:
                self.extra_clf = grid_search.best_estimator_
                y_pred = self.extra_clf.predict(self.X_test)  
                prev_mse = accuracy_score(self.y_test, y_pred)
                        
                print(i + 1, ". ", "Extra_clf", prev_mse)
            else:
                current_clf = ExtraTreesClassifier(n_estimators=grid_search.best_params_['n_estimators'])
                current_clf.fit(self.X_train, self.y_train)
                y_pred = current_clf.predict(self.X_test)  
                mse = accuracy_score(self.y_test, y_pred)
                        
                print(i + 1, ". ", "Extra_clf", mse)
                        
                if mse > prev_mse:
                    self.extra_clf = current_clf
                    prev_mse = mse
        
            i = i + 1
            
        self.clf_models.append( ("extra_clf" , self.extra_clf) )
        
        
    def rnd_compile_fit(self):
        param_grid = [ {"n_estimators" : [20,30,50, 70, 100,150,170,200,220], "max_depth":[10,15,20,40,50,60 ]}]
            
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid,  scoring="neg_mean_squared_error", verbose=2)
        grid_search.fit(self.X_train, self.y_train)
        
        print(grid_search.best_params_)
            
        y_pred = grid_search.predict(self.X_test)  
        mse = accuracy_score(self.y_test, y_pred)
        print(mse)      
                   
            
        prev_mse = 0     
        i = 0
        while(i < self.n_repetition):
                
            if i == 0:
                self.rnd_clf = grid_search.best_estimator_
                y_pred = self.rnd_clf.predict(self.X_test)  
                prev_mse = accuracy_score(self.y_test, y_pred)
                    
                print(i + 1 , ". ", "Rnd_clf", prev_mse)
            else:
                current_clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'])
                current_clf.fit(self.X_train, self.y_train)
                y_pred = current_clf.predict(self.X_test)  
                mse = accuracy_score(self.y_test, y_pred)
                    
                print(i + 1 , ". ", "Rnd_clf", mse)                
                if mse > prev_mse:
                    self.rnd_clf = current_clf
                    prev_mse = mse
    
            i = i + 1
            
        self.clf_models.append( ("rnd_clf" , self.rnd_clf) )
            

    
    def voting_compile_fit(self):
        i = 0
        while(i < self.n_repetition):
                
            if i == 0:
                self.voting_clf = VotingClassifier(estimators=self.clf_models, voting='soft')
                self.voting_clf.fit(self.X_train, self.y_train)
                y_pred = self.voting_clf.predict(self.X_test)  
                

                prev_acc = accuracy_score(self.y_test, y_pred)
                    
                print(i + 1 , ". ", "Voting_clf", prev_acc)
                    
            else:
                current_clf = VotingClassifier(estimators=self.clf_models, voting='soft')
                current_clf.fit(self.X_train, self.y_train)
                y_pred = current_clf.predict(self.X_test)  
                acc = accuracy_score(self.y_test, y_pred)
                    
                print(i + 1, ". ", "Voting_clf", acc)
                    
                if acc > prev_acc:
                    self.voting_clf = current_clf
                    prev_acc = acc
    
            i = i + 1
    
        