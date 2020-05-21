import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')

    data = data.drop(columns=['Ticket', 'Cabin']) # Let's try next without dropping 'Cabin'
    imputer = SimpleImputer(strategy='median')
    ordinal_enc = OrdinalEncoder()
    data_cat = data[['Sex']]
    data['Sex'] = ordinal_enc.fit_transform(data[['Sex']]).astype('int')
    print(data.info())
    # data_num = data.drop(columns=['Name'])
    # data['Age'].fillna(data['Age'].median(), inplace=True)
    # print(data['Ticket'].unique())
