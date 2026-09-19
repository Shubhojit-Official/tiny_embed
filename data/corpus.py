import numpy as np


def tokenize(text):
    return text.split()


def create_training_data(corpus, window_size=1):
    X = []
    Y = []

    for i, target_word in enumerate(corpus):

        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)

        for j in range(start, end):

            if j == i:
                continue

            X.append(target_word)
            Y.append(corpus[j])

    return np.array(X), np.array(Y)