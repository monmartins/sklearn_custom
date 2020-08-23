from sklearn.base import BaseEstimator, TransformerMixin

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