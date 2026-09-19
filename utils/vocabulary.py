class Vocabulary:
    def __init__(self, words):
        self.vocab = sorted(set(words))

        self.word_to_id = {
            word: i for i,word in enumerate(self.vocab)
        }

        self.id_to_word = {
            i: word for word, i in  self.word_to_id.items()
        }


    @property
    def size(self):
        return len(self.vocab)

    def encode(self, words):
        return [self.word_to_id[word] for word in words]

    def decode(self, ids):
        return [self.id_to_word[id] for id in ids]

    