import time
import numpy as np

from DataLoad import DataLoad

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer


TAG = '[MAIN]'


def split_data(x: np.array, y: list) -> tuple[np.array, np.array, np.array, np.array]:
    ss = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(x, y))

    test_index = ss[0][0]
    train_index = ss[0][1]

    x_train, x_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    return x_train, x_test, y_train, y_test


def tfidf(corpus: [str]) -> np.array:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()

    return X.toarray()


def main():
    start = time.time()

    dl = DataLoad()
    X, y = dl.load_bbc()

    X = tfidf(X)

    x_train, x_test, y_train, y_test = split_data(X, y)

    # clf = svm.SVC()
    # clf.fit(x_train, y_train)

    clf = RandomForestClassifier(max_depth=15, random_state=0)
    clf.fit(X, y)

    predictions = clf.predict(x_test)

    print(f'{TAG} CA: {round(accuracy_score(predictions, y_test), 4)}')
    print(f'{TAG} Time: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main()
