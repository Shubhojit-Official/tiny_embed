import numpy as np

from data.corpus import tokenize, create_training_data
from utils.vocabulary import Vocabulary
from models.numpy.skipgram import SkipGram


#------------Corpus------------
text = """
the cat sat on the mat
the cat ate the food
the cat likes the milk
the dog sat on the mat
the dog ate the food
the dog likes the milk
the dog chased the cat
the cat chased the mouse
the mouse ate the food
the mouse likes the cheese
the boy likes the dog
the boy chased the cat
the girl likes the cat
the girl likes the dog
"""

corpus = tokenize(text)

print("Number of words:", len(corpus))


# 2.------------Build Vocabulary------------

vocab = Vocabulary(corpus)

print("\nVocabulary:")
print(vocab.word_to_id)

V = len(vocab.word_to_id)

print("\nVocabulary size:", V)



# ------------Create Skip-Gram Training Data------------

window_size = 2

X_words, Y_words = create_training_data(
    corpus,
    window_size=window_size
)

print("\nNumber of training pairs:", len(X_words))

print("\nFirst 10 training pairs:")

for x, y in zip(X_words[:10], Y_words[:10]):
    print(f"{x} -> {y}")



#------------Convert Words -> IDs------------

X = np.array([
    vocab.word_to_id[word]
    for word in X_words
])

Y = np.array([
    vocab.word_to_id[word]
    for word in Y_words
])


print("\nFirst 10 ID pairs:")

for x, y in zip(X[:10], Y[:10]):
    print(f"{x} -> {y}")


#------------Create Skip-Gram Model------------

embedding_dim = 10

model = SkipGram(
    vocab_size=V,
    embedding_dim=embedding_dim
)


#------------Train------------

epochs = 500
learning_rate = 0.05

losses = model.train(
    X,
    Y,
    epochs=epochs,
    learning_rate=learning_rate
)


#------------Inspect Embeddings------------

embeddings = model.get_embeddings()

print("\nFinal Embeddings:\n")

for word, word_id in vocab.word_to_id.items():
    print(f"{word:>10} : {embeddings[word_id]}")


#------------Cosine Similarity------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def most_similar(word, top_k=5):

    if word not in vocab.word_to_id:
        print(f"'{word}' is not in the vocabulary.")
        return

    word_id = vocab.word_to_id[word]

    target_embedding = embeddings[word_id]

    similarities = []

    for other_word, other_id in vocab.word_to_id.items():

        if other_word == word:
            continue

        similarity = cosine_similarity(
            target_embedding,
            embeddings[other_id]
        )

        similarities.append(
            (other_word, similarity)
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nWords most similar to '{word}':")

    for other_word, similarity in similarities[:top_k]:

        print(
            f"{other_word:>10} : {similarity:.4f}"
        )


#------------Test Embeddings------------

most_similar("cat")
most_similar("dog")
most_similar("boy")