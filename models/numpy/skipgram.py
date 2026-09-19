import numpy as np

class SkipGram:

    def __init__(self, vocab_size, embedding_dim):
        """
        Skip-gram model using full softmax.

        Parameters
        ----------
        vocab_size : int
            Number of words/tokens in the vocabulary.

        embedding_dim : int
            Number of dimensions in each word embedding.
        """

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        #Input Embedding Matrix
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        #Output Embedding Matrinx
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01


    #--------Activation-------
    @staticmethod
    def softmax(z):
        """
        Convert logits into probabilities.
        """

        # Subtracting max(z) improves numerical stability.
        exp_z = np.exp(z - np.max(z))

        return exp_z / np.sum(exp_z)

    #------Forward Pass + Backward + Update-------------

    def train_step(
        self,
        input_id,
        target_id,
        learning_rate
    ):
        """
        Perform one complete training step.

        Forward pass
            ↓
        Loss calculation
            ↓
        Backpropagation
            ↓
        Parameter update
        """

        # -------Forward Pass-----------
        
        # Get Embedding for the input
        h = self.W_in[input_id]

        # Calculate Logits
        z = h @ self.W_out
        
        # Softmax
        p = self.softmax(z)

        # Loss function (Cross-Entropy)
        loss = -np.log(p[target_id])

        # ------Back Propagation--------
        
        # Gradient of loss wrt logits (dL/dz)
        dz = p.copy()
        dz[target_id] -= 1
    
        # Gradient for output matrix  (dL/dWout)
        dWout =  np.outer(h, dz) # (h{trans} * p - y )
    
        # Gradient for the Embedding (dL/dh)
        dh = dz @ self.W_out.T
    
        #---------Updation----------
        self.W_out -= learning_rate * dWout
        self.W_in[input_id] -= learning_rate * dh
    
        return loss

    def train(
        self,
        X,
        y,
        epochs=100,
        learning_rate=0.05,
        verbose=True
    ):
        """
        Train the Skip-gram model.

        X:
            Input word IDs.

        y:
            Target/context word IDs.
        """

        history = []

        for epoch in range(epochs):

            total_loss = 0.0

            for input_id, target_id in zip(X, y):

                loss = self.train_step(
                    input_id=input_id,
                    target_id=target_id,
                    learning_rate=learning_rate
                )

                total_loss += loss

            average_loss = total_loss / len(X)

            history.append(average_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Loss: {average_loss:.4f}"
                )

        return history

    #------Embedding access------

    def get_embedding(self, word_id):
        """
        Return the learned embedding for a word ID.
        """

        return self.W_in[word_id]

    def get_embeddings(self):
        """
        Return the complete embedding matrix.
        """

        return self.W_in