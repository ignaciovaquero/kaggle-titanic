import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report


class Titanic:
    def __init__(self, train_data, test_data):
        self.data = pd.read_csv(train_data)
        self.test = pd.read_csv(test_data)
        self.data_cleaned = self.clean_data(self.data)
        self.labels = self._extract_labels(self.data)
        self.names = self._extract_names(self.data)
        self.gb = GradientBoostingClassifier(n_estimators=200)


    def _extract_labels(self, data):
        return data.dropna(subset=['Embarked'])['Survived'].to_numpy()

    def _extract_names(self, data):
        return data.dropna(subset=['Embarked'])['Name'].to_numpy()


    def clean_data(self, data):
        '''
        Clean and transform Titanic Data
        '''

        # Let's try next without dropping 'Cabin'
        data = data.dropna(subset=['Embarked'])
        data = data[['Pclass', 'Sex', 'Age',
                     'SibSp', 'Parch', 'Fare', 'Embarked']]
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
        return data_cleaned


    def fit(self):
        self.gb.fit(self.data_cleaned, self.labels)


    def predict(self, data):
        names = self._extract_names(data)
        y = self.gb.predict(self.clean_data(data))
        predicted = pd.DataFrame(data=y, columns=['Survived'])
        predicted['Name'] = names
        return predicted

    def confusion_matrix(self, predicted):
        return confusion_matrix(self.labels, predicted)
