import numpy as np

from models import *
from helpers import *

SEED = 5


def ridge_regression():
    train_vals, train_lbls, train_cols = read_data('train.csv')
    valid_vals, valid_lbls, valid_cols = read_data('validation.csv')
    test_vals, test_lbls, test_cols = read_data('test.csv')

    lambd_set = np.array([0, 2, 4, 6, 8, 10])
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

    accuracies = {'Lambda': lambd_set, 'Train Accuracy': train_acc,
                  'Validation Accuracy': valid_acc, 'Test Accuracy': test_acc}
    df = pd.DataFrame(accuracies, index=None)

    # Q6.1
    # Display accuracies DF
    print(df)

    # Plot train acc's
    plt.scatter(lambd_set, train_acc)
    plt.title('Train accuracies')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.show()

    # Plot validation acc's
    plt.scatter(lambd_set, valid_acc)
    plt.title('Validation accuracies')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.show()

    # Plot test acc's
    plt.scatter(lambd_set, test_acc)
    plt.title('Test accuracies')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.show()

    # Q6.2
    best_lambda_index = np.argmax(valid_acc)
    worst_lambda_index = np.argmin(valid_acc)

    # Plot visualization for best and worst models
    plot_decision_boundaries(models[best_lambda_index], test_vals, test_lbls,
                             title='Best lambda:{} - visualization plot'.format(lambd_set[best_lambda_index]))

    plot_decision_boundaries(models[worst_lambda_index], test_vals, test_lbls,
                             title='Worst lambda:{} - visualization plot'.format(lambd_set[worst_lambda_index]))


def gradient_descent():
    time_vector = np.linspace(1, 1000, 1)


if __name__ == '__main__':
    np.random.seed(SEED)
    ridge_regression()
