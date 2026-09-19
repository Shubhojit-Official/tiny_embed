import numpy as np

def one_hot(word_id, V):
    vector = np.zeros(V)
    vector[word_id] = 1
    return vector

def embedding(V, dim=3):
    E =  np.random.randn(V, dim)
    return E

def create_training_data(corpus, window_size = 1):
    X = []
    Y = []

    for i , target_word in enumerate(corpus):
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)

        for j in range(start, end):

            if j == i:
                continue

            X.append(target_word)
            Y.append(corpus[j])
    return np.array(X), np.array(Y)

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

    embed_mat = embedding(vocab_size, 3)

    X, y = create_training_data(corpus, window_size=1)

    for input_id, target_id in zip(X, y):
        print(
            id_to_word[input_id],
            "->",
            id_to_word[target_id]
        )

    