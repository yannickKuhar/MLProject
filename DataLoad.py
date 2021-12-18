import os
import nltk
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download()


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
