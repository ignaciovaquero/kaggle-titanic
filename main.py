import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def transform_data(data):
    '''
    Clean and transform Titanic Data
    '''

    # Let's try next without dropping 'Cabin'
    data = data.drop(columns=['PassengerId', 'Ticket',
                              'Cabin', 'Name', 'Survived'])
    data = data.dropna(subset=['Embarked'])
    data_num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    data_cat = ['Sex']
    data_1h = ['Embarked']
    column_transformer = ColumnTransformer([
        ("num", SimpleImputer(strategy='median'), data_num),
        ("cat", OrdinalEncoder(), data_cat),
        ("1h", OneHotEncoder(sparse=False), data_1h)
    ])

    array_processed = column_transformer.fit_transform(data)
    data_processed = pd.DataFrame(data=array_processed, columns=[
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'C', 'Q', 'S'
    ])
    data_float = data_processed[['Fare', 'Age']]
    data_processed = data_processed.drop(columns=['Fare', 'Age']).astype('int')
    data_processed[['Fare', 'Age']] = data_float
    scaler = StandardScaler()
    data_processed[['Fare', 'Age']] = scaler.fit_transform(
        data_processed[['Fare', 'Age']])
    return data_processed


def predict():
    pass


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')

    labels = data[['Name','Survived']]
    data = transform_data(data)
    print(data.head())
    print(labels.info())
