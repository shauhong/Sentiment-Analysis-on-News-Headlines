import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.tokenize import word_tokenize

class BiLSTM(nn.Module):
    def __init__(self, glove, n_classes=3, hidden_size=16, num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glove = glove
        self.lstm = nn.LSTM(input_size=glove.glove.vector_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=n_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        embeddings = self.glove(x)
        output, (hidden_state, cell_state) = self.lstm(embeddings) 
        final_hidden_state = torch.cat((hidden_state[-2], hidden_state[-1]), dim=-1)
        final_hidden_state = self.dropout(final_hidden_state)
        logits = self.fc(final_hidden_state)
        return logits
        
    def inference(self, x):
        assert isinstance(x, str)
        x = self.preprocess(x)
        logits = self.forward(x)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        return predictions
            
    def preprocess(self, x):
        x = x.lower()
        x = word_tokenize(x)
        x = [x]
        x = np.array(self.glove.words2indices(x))
        x = torch.t(torch.tensor(x, dtype=torch.long))
        x = x.to(self.device)
        return x
    
class AttentionBiLSTM(nn.Module):
    def __init__(self, glove, n_classes, hidden_size=16, num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glove = glove
        self.lstm = nn.LSTM(input_size=glove.glove.vector_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.attention = nn.Linear(in_features=2*hidden_size, out_features=2*hidden_size)
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=n_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        embeddings = self.glove(x)
        outputs, (hidden_state, cell_state) = self.lstm(embeddings)    
        final_hidden_state = torch.cat((hidden_state[-2], hidden_state[-1]), dim=-1) 
        attention_weights = self.attention(outputs).permute([1, 0, 2])
        attention_weights = torch.bmm(attention_weights, final_hidden_state.unsqueeze(2))
        attention_weights = F.softmax(attention_weights.squeeze(2), dim=-1)
        attention_context = torch.bmm(outputs.permute([1, 2, 0]), attention_weights.unsqueeze(2)).squeeze(2)
        attention_context = self.dropout(attention_context)
        logits = self.fc(attention_context)
        return logits
           
    def inference(self, x):
        assert isinstance(x, str)
        x = self.preprocess(x)
        logits = self.forward(x)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1) 
        return predictions
            
    def preprocess(self, x):
        x = x.lower()
        x = word_tokenize(x)
        x = [x]
        x = np.array(self.glove.words2indices(x))
        x = torch.t(torch.tensor(x, dtype=torch.long))
        x = x.to(self.device)
        return x