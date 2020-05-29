from titanic import Titanic
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    t = Titanic(train_data='data/train.csv', test_data='data/test.csv')
    t.fit()
    y_predict = t.predict(t.data)
    y_label = t.labels
    print(confusion_matrix(y_label, y_predict['Survived']))
    print(classification_report(y_label, y_predict['Survived']))

