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

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def train_step(input_id, target_id, W_in, W_out, learning_rate):
    # -------Forward Pass-----------

    # Get Embedding for the input
    h = W_in[input_id]

    # Calculate Logits
    z = h @ W_out
    
    # Softmax
    p = softmax(z)

    # Loss function (Cross-Entropy)
    loss = -np.log(p[target_id])

    # ------Back Propagation--------

    # Gradient of loss wrt logits (dL/dz)
    dz = p.copy()
    dz[target_id] -= 1

    # Gradient for output matrix  (dL/dWout)
    dWout =  np.outer(h, dz) # (h{trans} * p - y )

    # Gradient for the Embedding (dL/dh)
    dh = dz @ W_out.T

    #---------Updation----------
    W_out -= learning_rate * dWout
    W_in[input_id] -= learning_rate * dh

    return loss


if __name__ == "__main__":
    text = "the cat sat on the mat"
    words  = text.split()

    vocab = sorted(set(words))
    vocab_size = len(vocab)
    embedding_dim = 3

    word_to_id = {
        word: i for i,word in enumerate(vocab)
    }

    id_to_word = {
        i : word for word, i in word_to_id.items()
    }

    corpus = [word_to_id[word] for word in words]


    np.random.seed(42)

    W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
    W_out = np.random.randn(embedding_dim, vocab_size) * 0.01

    X, y = create_training_data(corpus,1)

    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for input_id, target_id in zip(X, y):
            loss = train_step(
                input_id,
                target_id,
                W_in,
                W_out,
                learning_rate=0.1
            )
            total_loss += loss

        average_loss = total_loss / len(X)
        if epoch % 10 == 0:
            print(
            f"Epoch {epoch}, "
            f"Loss: {average_loss:.4f}"
            )

    for word in vocab:
        word_id = word_to_id[word]

        print(
            word,
            W_in[word_id]
        )

  
    