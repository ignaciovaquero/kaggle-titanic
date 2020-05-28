from titanic import Titanic


if __name__ == "__main__":
    t = Titanic(train_data='data/train.csv', test_data='data/test.csv')
    t.fit()
    y_predict = t.predict(t.data)
    y_label = t.labels
    print(y_predict[:5])
    print(y_label[:5])
