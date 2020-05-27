from titanic import Titanic

if __name__ == "__main__":
    t = Titanic(train_data='data/train.csv', test_data='data/test.csv')
    t.fit()
