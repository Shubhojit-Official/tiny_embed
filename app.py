import numpy as np

def one_hot(word_id, V):
    vector = np.zeros(V)
    vector[word_id] = 1
    return vector


if __name__ == "__main__":
    text = "the cat sat on the mat"
    words  = text.split()

    vocab = sorted(set(words))
    vocab_size = len(vocab)

    word_to_id = {
        word: i for i,word in enumerate(vocab)
    }

    id_to_word = {
        i : word for word, i in word_to_id.items()
    }

    corpus = [word_to_id[word] for word in words]

    print(one_hot(word_to_id['on'], vocab_size))