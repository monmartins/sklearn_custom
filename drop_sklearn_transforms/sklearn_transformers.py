from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

class RemoveZerosRows(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        
        ignoredColumns = data.columns.tolist()
        ignoredColumns = [i for i in ignoredColumns if i not in self.columns]
        data_ignored = data[ignoredColumns]
        
        data.dropna(subset=self.columns, inplace=True)
        data = data[data[self.columns] != 0]
        data[ignoredColumns] = data_ignored
        
        return data

class LabelEncoderColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label = preprocessing.LabelEncoder()
        self.encode = True

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        if(self.encode == True):
            data[self.columns] = self.label.fit_transform(data[self.columns])
            self.encode = False
        else:
            data[self.columns] = self.label.inverse_transform(data[self.columns])
            self.encode = True
        return data

class RandomForestRegressorCustom(BaseEstimator, TransformerMixin):
    def __init__(self, columns, max_depth=5, random_state=0):
        self.columns = columns
        self.max_depth = max_depth
        self.random_state = random_state
        self.label = preprocessing.LabelEncoder() 
        self.regr = RandomForestRegressor(self.max_depth, self.random_state)

    def fit(self, X, y=None):
        
        if y != None:
            y2_train = self.label.fit_transform(y) 
        else:
            y2_train = self.label.fit_transform(X[self.columns]) 
            
        self.regr.fit(X,y2_train)
        return self.regr
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        
        return data
    def predict(self, X):
        y_pred = self.regr.predict(X)
        return self.label.inverse_transform(y_pred)

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')