from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

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

class LabelEncoderCustom(BaseEstimator, TransformerMixin):
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