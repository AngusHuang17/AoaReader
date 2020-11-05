class Dictionary:

    def __init__(self, word2id):
        self.word2id = word2id
        self.id2word = {id: word for word, id in word2id.items()}
        self.len = len(self.word2id)

    def getWord(self, id):
        return self.id2word[id]

    def getId(self, word):
        return self.word2id[word]

    def Convert2Words(self, ids):
        return [self.getWord(id) for id in ids]

    def Convert2Ids(self, words):
        return [self.getId(word) for word in words]