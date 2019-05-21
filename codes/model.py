import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, rnn_cell, input_dim, embedding_dim, hidden_dim, output_dim,device):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = rnn_cell(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, text):
        # TODO: your codes here
        # text: [sent len, batch size]
        
        # 1. get embdding vectors
        # embedded: [sent len, batch size, emb dim]
        embedded = self.embedding(text)

        # 2. initialize hidden vector (considering special parts of LSTMCell)
        # hidden: [1, batch size, hid dim]
        if self.rnn.__class__.__name__ == "LSTMCell":
            hidden = (
                torch.zeros(embedded.shape[1], self.hidden_dim).view(-1, self.hidden_dim),
                torch.zeros(embedded.shape[1], self.hidden_dim).view(-1, self.hidden_dim)).to(self.device)
        else:
            # hidden = torch.zeros(embedded.shape[1], self.hidden_dim).view(1,-1, self.hidden_dim)
            hidden = torch.zeros(embedded.shape[1], self.hidden_dim).view(-1, self.hidden_dim).to(self.device)

        # 3. multiple step recurrent forward
        for step in range(embedded.shape[0]):
            hidden = self.rnn(embedded[step],hidden)

        # 4. get final output
        if self.rnn.__class__.__name__ == "LSTMCell":
            return self.fc(hidden[0])
        else:
            return self.fc(hidden)
