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

    X, y = create_training_data(corpus, window_size=1)

    embed_mat = embedding(vocab_size, embedding_dim)

    W_in = embed_mat * 0.01 # Input Embedding Matrix
    W_out = np.random.randn(embedding_dim, vocab_size) * 0.01


    input_id = word_to_id['cat']
    target_id = word_to_id['sat']

    # Get Embedding for the input
    h = W_in[input_id]

    # Calculate Logits
    z = h @ W_out

    # Softmax
    p = softmax(z)

    # Loss function (Cross-Entropy)
    loss = -np.log(p[target_id])

    # Gradient of loss wrt logits
    dz = p.copy()
    dz[target_id] -= 1

    # Gradient for output matrix (dL/dW_out = h(transpose)*(p-y))
    dW_out =  np.outer(h,dz)

    # Gradient for embedding
    dh = dz @ W_out.T
    
  
    