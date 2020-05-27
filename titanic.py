import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


class Titanic:
    def __init__(self, train_data, test_data):
        self.data = pd.read_csv(train_data)
        self.test = pd.read_csv(test_data)
        self.data_cleaned, self.labels = self.clean_data(self.data)
        self.gb = GradientBoostingClassifier(n_estimators=200)


    def clean_data(self, data):
        '''
        Clean and transform Titanic Data
        '''

        # Let's try next without dropping 'Cabin'
        data = data.dropna(subset=['Embarked'])
        labels = data[['Survived']]
        data = data.drop(columns=['PassengerId', 'Ticket',
                                  'Cabin', 'Name', 'Survived'])
        data_num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        data_cat = ['Sex']
        data_1h = ['Embarked']
        column_transformer = ColumnTransformer([
            ("num", SimpleImputer(strategy='median'), data_num),
            ("cat", OrdinalEncoder(), data_cat),
            ("1h", OneHotEncoder(sparse=False), data_1h)
        ])

        array_processed = column_transformer.fit_transform(data)
        data_cleaned = pd.DataFrame(data=array_processed, columns=[
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'C', 'Q', 'S'
        ])
        data_float = data_cleaned[['Fare', 'Age']]
        data_cleaned = data_cleaned.drop(columns=['Fare', 'Age']).astype('int')
        data_cleaned[['Fare', 'Age']] = data_float
        scaler = StandardScaler()
        data_cleaned[['Fare', 'Age']] = scaler.fit_transform(
            data_cleaned[['Fare', 'Age']])
        return data_cleaned,labels


    def fit(self):
        self.gb.fit(self.data_cleaned, self.labels)


    def predict(self, data):
        return self.gb.predict(self.clean_data(data))
