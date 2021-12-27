import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud


class WC:
    def __init__(self, data, predictions, n_words):
        self.data = np.array(data)
        self.predictions = predictions
        self.n_words = n_words

    def split_by_class(self):
        split = {p: [] for p in set(self.predictions)}

        for x, p in zip(self.data, self.predictions):
            split[p].append(x)

        return split

    def most_common_n_words(self, corpus):
        word_dict = dict()

        for doc in corpus:
            for word in doc:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

        most_common_words = sorted(word_dict, key=word_dict.get, reverse=True)

        return most_common_words[:self.n_words]

    def show_word_cloud(self, words):
        cloud = WordCloud(max_font_size=50, background_color='white').generate(' '.join(words))

        plt.figure()
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def run(self):
        split = self.split_by_class()

        for key in split:
            print(f'Key: {key}')
            common_words = self.most_common_n_words(split[key])
            self.show_word_cloud(common_words)
