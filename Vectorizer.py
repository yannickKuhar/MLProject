import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Vectorizer:

    @staticmethod
    def tfidf(corpus):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

        return X.toarray()

    @staticmethod
    def d2vec(x_train, x_test):
        vec_size = int(sum([len(doc) for doc in x_train]) / len(x_train))

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(x_train)]

        model = Doc2Vec(documents, vector_size=vec_size)

        return np.array([model.infer_vector(doc) for doc in x_test])
