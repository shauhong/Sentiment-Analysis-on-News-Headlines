import torch
import numpy as np
import gensim.downloader as api

class GloVe(object):
    def __init__(self, variant="glove-wiki-gigaword-300"):
        self.variant = variant

        self.word2id = dict()
        self.id2word = dict()
        
        self.word2embedding = dict()
        self.id2embedding = None
        
        self.word2id['<pad>'] = 0
        self.word2id['<unk>'] = 1
        
        self.word2embedding['<pad>'] = None
        self.word2embedding['<unk>'] = None
       
        self.unknown_idx = self.word2id['<unk>']
        self.padding_idx = self.word2id['<pad>']
        
        self.load()
        
    def __len__(self):
        return len(self.word2id)
    
    def __getitem__(self, indices):
        return self.id2embedding[indices]
    
    def __call__(self, indices):
        return self.id2embedding[indices]
    
    def indices2words(self, indices):
        return [self.id2word[index] for index in indices]
    
    def words2indices(self, sentences):
        if type(sentences[0]) == list:
            return [[self.word2id.get(w, self.unknown_idx) for w in sentence] for sentence in sentences]
        else:
            return [self.word2id.get(w, self.unknown_idx) for w in sentences]
    
    def load(self):
        self.glove = api.load(self.variant)
        embedding_size = self.glove.vector_size
        for word in self.glove.index_to_key:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
                self.word2embedding[word] = self.glove[word]
        self.word2embedding['<unk>'] = np.mean([self.glove[word] for word in self.glove.index_to_key], axis=0)
        self.word2embedding['<pad>'] = np.zeros(embedding_size)
        self.id2word = {v:k for k,v in self.word2id.items()}
        self.id2embedding = torch.tensor([self.word2embedding[word] for index, word in sorted(self.id2word.items())], dtype=torch.float)