import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')

    labels = data[['Name','Survived']]
    # Let's try next without dropping 'Cabin'
    data = data.drop(columns=['Ticket', 'Cabin', 'Name','Survived'])
    data = data.dropna(subset=['Embarked'])
    data_num = data.drop(columns=['Embarked'])
    data_cat_1h = data['Embarked']
    imputer = SimpleImputer(strategy='median')
    ordinal_enc = OrdinalEncoder()
    data_cat = data[['Sex']]
    data['Sex'] = ordinal_enc.fit_transform(data[['Sex']]).astype('int')
    print(data.info())
    print(labels.head())
    # data_num = data.drop(columns=['Name'])
    # data['Age'].fillna(data['Age'].median(), inplace=True)
    # print(data['Ticket'].unique())
