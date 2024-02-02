from models import *
from helpers import *

def main():
    train_vals, train_lbls, train_cols = read_data('train.csv')
    valid_vals, valid_lbls, valid_cols = read_data('validation.csv')
    test_vals, test_lbls, test_cols = read_data('test.csv')

    lambd_set = [0,2,4,6,8,10]
    models = np.empty(len(lambd_set), dtype=object)
    train_acc = np.zeros(len(lambd_set))
    valid_acc = np.zeros(len(lambd_set))
    test_acc = np.zeros(len(lambd_set))


    for i, lambd in enumerate(lambd_set):
        model = Ridge_Regression(lambd)
        model.fit(train_vals, train_lbls)
        train_acc[i] = np.mean(model.predict(train_vals) == train_lbls)
        valid_acc[i] = np.mean(model.predict(valid_vals) == valid_lbls)
        test_acc[i] = np.mean(model.predict(test_vals) == test_lbls)
        models[i] = model




if __name__ == '__main__':
    pass