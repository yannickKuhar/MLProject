import os
import nltk
import keras
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# nltk.download()


from keras.datasets import imdb


class DataLoad:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = stopwords.words('english')
        self.tag = '[DATA_LOAD]'

    def process_text(self, article):
        # Convert article to lower case tokens.
        tokens = nltk.word_tokenize(article, 'english')

        # Strip punctuations from each lower case word.
        table = str.maketrans('', '', string.punctuation)
        stripped = [word.lower().translate(table) for word in tokens]

        # Remove non alphabetic tokens and stop words, then stem the remaining.
        return [self.stemmer.stem(word) for word in stripped if word.isalpha() and word not in self.stop_words]

    def load_bbc_folder(self, folder: str, class_encoding: int, corpus: list):
        with os.scandir('Data/bbc/' + folder) as entries:
            for text in entries:
                f = open(text, 'r', encoding='unicode_escape')

                words = self.process_text(f.read().strip())

                corpus.append((' '.join(words), class_encoding))
                f.close()

    def load_bbc(self):
        data = []

        self.load_bbc_folder('business', 0, data)
        self.load_bbc_folder('entertainment', 1, data)
        self.load_bbc_folder('politics', 2, data)
        self.load_bbc_folder('sport', 3, data)
        self.load_bbc_folder('tech', 4, data)

        np.random.shuffle(data)

        X = []
        y = []

        for point in data:
            X.append(point[0])
            y.append(point[1])

        return X, y

    def load_emails_folder(self, folder, class_encoding, data):
        with os.scandir(r'Data/emails/' + folder) as entries:
            for text in entries:
                try:
                    f = open(text, 'r', encoding='unicode_escape')

                    words = self.process_text(f.read().strip())

                    data.append((' '.join(words), class_encoding))

                    f.close()
                except:
                    pass

    def load_emails(self):
        data = []

        self.load_emails_folder('ham', 0, data)
        self.load_emails_folder('spam', 1, data)

        np.random.shuffle(data)

        X = []
        y = []

        for point in data:
            X.append(point[0])
            y.append(point[1])

        return X, y

    def process_imdb(self, texts, inverted_word_index):
        print(f'{self.tag} Formatting IMDB data start.')

        data = []

        for seq in texts:
            decoded_sequence = ' '.join(inverted_word_index[i] for i in seq)
            decoded_sequence_processed = ' '.join(self.process_text(decoded_sequence))
            data.append(decoded_sequence_processed)

        print(f'{self.tag} Formatting IMDB data done.')
        return data

    def load_imdb(self):
        print(f'{self.tag} IMDB load start.')

        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

        np.load = np_load_old

        word_index = keras.datasets.imdb.get_word_index()
        inverted_word_index = dict((i, word) for (word, i) in word_index.items())

        x_train_decoded = self.process_imdb(x_train, inverted_word_index)
        x_test_decoded = self.process_imdb(x_test, inverted_word_index)

        print(f'{self.tag} IMDB load done.')

        return x_train_decoded, x_test_decoded, y_train, y_test
