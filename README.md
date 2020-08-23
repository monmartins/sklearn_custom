# sklearn_custom


This Python package encapsulates custom sklearn pipeline transforms for use with the Watson Machine Learning API



## Installation

        git clone https://github.com/monmartins/sklearn_custom
        cd sklearn_custom
        ls -ltr
        zip -r sklearn_transforms.zip sklearn_transforms
        pip install sklearn_transforms.zip
## Code example
```python

    from drop_sklearn_transforms.sklearn_transformers import DropColumns, RemoveZerosRows 

    rm_columns = DropColumns(
        columns=["COLUMN1"]  # Essa transformação recebe como parâmetro uma lista com os nomes das colunas indesejadas
    )

    # Aplicando a transformação ``DropColumns`` ao conjunto de dados base
    rm_columns.fit(X=df_data_1)

    # Reconstruindo um DataFrame Pandas com o resultado da transformação
    df_data_2 = pd.DataFrame.from_records(
        data=rm_columns.transform(
            X=df_data_1
        ),
    )
```
