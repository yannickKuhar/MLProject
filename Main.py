import sys
import time
import numpy as np

from WC import WC
from DataLoad import DataLoad
from Vectorizer import Vectorizer

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

TAG = '[MAIN]'
ERROR = '[ERROR]'

SVM = 'svm'
RF = 'rf'
NB = 'nb'

BBC = 'bbc'
EMAILS = 'emails'
IMDB = 'imdb'

TDIDF = 'tdidf'
D2V = 'd2v'


def split_data(x, y):
    ss = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(x, y))

    test_index = ss[0][0]
    train_index = ss[0][1]

    x_train, x_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    return x_train, x_test, y_train, y_test, train_index, test_index


def main(argv):
    start = time.time()

    X = []
    y = []
    dl = DataLoad()
    vec = Vectorizer()

    x_train, x_test, y_train, y_true, y_test = None, None, None, None, None

    pre_embedding_data = None

    if argv[2] == BBC:
        X, y = dl.load_bbc()
    elif argv[2] == EMAILS:
        X, y = dl.load_emails()
    elif argv[2] == IMDB:
        x_train, x_test, y_train, y_true = dl.load_imdb()

        if argv[3] == TDIDF:
            pre_embedding_data = [x.split(' ') for x in x_test]
            x_train = vec.tfidf(x_train)
            x_test = vec.tfidf(x_test)

        if argv[3] == D2V:
            x_train = [x.split(' ') for x in x_train]
            x_test = [x.split(' ') for x in x_test]
            x_test_pre = np.array(x_test)

            X = vec.d2vec(x_train, x_test)
            x_train, x_test, y_train, y_test, train_index, test_index = split_data(X, y_true)
            pre_embedding_data = x_test_pre[test_index]
    else:
        print(f'{TAG} {ERROR} Invalid data.')
        return -1

    if argv[3] == TDIDF and argv[2] != IMDB:
        X_pre = np.array([x.split(' ') for x in X])
        X = vec.tfidf(X)
        x_train, x_test, y_train, y_test, train_index, test_index = split_data(X, y)
        pre_embedding_data = X_pre[test_index]

    if argv[3] == D2V and argv[2] != IMDB:
        X = [x.split(' ') for x in X]
        x_train, x_test, y_train, y_true, _, _ = split_data(X, y)
        x_test_pre = x_test

        X = vec.d2vec(x_train, x_test)
        x_train, x_test, y_train, y_test, train_index, test_index = split_data(X, y_true)
        pre_embedding_data = x_test_pre[test_index]

    if argv[1] == SVM:
        clf = svm.SVC()
    elif argv[1] == RF:
        clf = RandomForestClassifier(max_depth=15, random_state=0)
    elif argv[1] == NB:
        clf = GaussianNB()
    else:
        print(f'{TAG} {ERROR} Invalid ML model.')
        return -1

    ######################################## Get predictions. ########################################
    print(f'{TAG} Model: {argv[1]} training start.')
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)

    if argv[2] == IMDB and argv[3] == TDIDF:
        print(f'{TAG} CA: {round(accuracy_score(predictions, y_true), 2)}')
    else:
        print(f'{TAG} CA: {round(accuracy_score(predictions, y_test), 2)}')
    ##################################################################################################

    ######################################## Get word cloud. ########################################

    wc = WC(pre_embedding_data, predictions, 200)
    wc.run()

    #################################################################################################

    print(f'{TAG} Time: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main(sys.argv)
