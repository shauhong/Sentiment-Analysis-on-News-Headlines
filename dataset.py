import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import os
from utils import pad_sentences

class News(Dataset):
    def __init__(self, path, glove, split="train"):
        self.path = os.path.join(path, split + '.csv')
        self.glove = glove
        self.df = pd.read_csv(self.path, header=None)
    
    def __getitem__(self, index):
        sentence, sentiment = self.df.iloc[index]
        sentence, sentiment = self.transform(sentence, sentiment)
        return sentence, sentiment
    
    def __len__(self):
        return len(self.df)
    
    def transform(self, sentence, sentiment):
        sentence = sentence.lower()
        sentence = word_tokenize(sentence)
        sentiment += 1
        return sentence, sentiment
    
    def collate_fn(self, batch):
        sentences = []
        sentiments = []
        for sentence, sentiment in batch:
            sentences.append(sentence)
            sentiments.append(sentiment)
        sentences_padded = pad_sentences(sentences, pad_token='<pad>')
        sentences_padded = np.array(self.glove.words2indices(sentences_padded))
        sentiments = np.array(sentiments)
        sentences_padded = torch.t(torch.tensor(sentences_padded, dtype=torch.long))
        sentiments = torch.tensor(sentiments, dtype=torch.long)
        return sentences_padded, sentiments