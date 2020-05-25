import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')

    labels = data[['Name','Survived']]
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

    pipeline = Pipeline([("cleaning", column_transformer), ("scaling", StandardScaler())])
    array_processed = pipeline.fit_transform(data)
    data_processed = pd.DataFrame(data=array_processed, columns=[
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'C', 'Q', 'S'
        ])
    # data_float = data_processed['Fare']
    # data_processed = data_processed.drop(columns='Fare').astype('int')
    # data_processed['Fare'] = data_float
    print(data_processed.info())
    print(labels.info())
