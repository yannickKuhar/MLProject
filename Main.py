import sys
import time
import numpy as np

from DataLoad import DataLoad
from Vectorizer import Vectorizer

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

TAG = '[MAIN]'
ERROR = '[ERROR]'

SVM = 'svm'
RF = 'rf'
NB = 'nb'

BBC = 'bbc'
EMAILS = 'emails'
IMDB = 'imdb'


def split_data(x, y):
    ss = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(x, y))

    test_index = ss[0][0]
    train_index = ss[0][1]

    x_train, x_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    return x_train, x_test, y_train, y_test


def main(argv):
    start = time.time()

    X =[]
    y = []
    dl = DataLoad()

    if argv[2] == BBC:
        X, y = dl.load_bbc()
        X = [x.split(' ') for x in X]
    elif argv[2] == EMAILS:
        X, y = dl.load_emails()
        X = [x.split(' ') for x in X]
    elif argv[2] == IMDB:
        x_train, x_test, y_train, y_test = dl.load_imdb()
    else:
        print(f'{TAG} {ERROR} Invalid data.')
        return -1

    if argv[2] != IMDB:
        x_train, x_test, y_train, y_test = split_data(X, y)

    vec = Vectorizer()
    X = vec.d2vec(x_train, x_test)

    # print(X)

    return -1

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

    print(f'{TAG} CA: {round(accuracy_score(predictions, y_test), 4)}')
    print(f'{TAG} Time: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main(sys.argv)
    # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    #
    # print(common_texts)
    # print(len(common_texts))
    # # print(type(common_texts), type(common_texts[0]), type(common_texts[0][0]))
    #
    # model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    # vecs = model.infer_vector(common_texts[0])
    #
    # print(vecs)

