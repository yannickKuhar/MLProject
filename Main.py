import sys
import time
import numpy as np

from DataLoad import DataLoad

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

TAG = '[MAIN]'
ERROR = '[ERROR]'

SVM = 'svm'
RF = 'rf'
NB = 'nb'

BBC = 'bbc'
EMAILS = 'emails'
IMDB = 'imdb'


def split_data(x: np.array, y: list):
    ss = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(x, y))

    test_index = ss[0][0]
    train_index = ss[0][1]

    x_train, x_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    return x_train, x_test, y_train, y_test


def tfidf(corpus: [str]) -> np.array:
    print(f'{TAG} TDIDF start.')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    print(f'{TAG} TDIDF done.')

    return X.toarray()


def main(argv):
    start = time.time()

    dl = DataLoad()

    if argv[2] == BBC:
        X, y = dl.load_bbc()
    elif argv[2] == EMAILS:
        X, y = dl.load_emails()
    elif argv[2] == IMDB:
        x_train, x_test, y_train, y_test = dl.load_imdb()
        x_train = tfidf(x_train)
        x_test = tfidf(x_test)
    else:
        print(f'{TAG} {ERROR} Invalid data.')
        return -1

    if argv[2] != IMDB:
        X = tfidf(X)
        x_train, x_test, y_train, y_test = split_data(X, y)

    if argv[1] == SVM:
        clf = svm.SVC()
    elif argv[1] == RF:
        clf = RandomForestClassifier(max_depth=15, random_state=0)
    elif argv[1] == NB:
        clf = GaussianNB()
    else:
        print(f'{TAG} {ERROR} Invalid ML model.')
        return -1

    print(f'{TAG} Model: {argv[1]} training start.')
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(f'{TAG} Model: {argv[1]} training done.')

    print(f'{TAG} CA: {round(accuracy_score(predictions, y_test), 4)}')
    print(f'{TAG} Time: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main(sys.argv)
