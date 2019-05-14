import torch.nn as nn


class Model(nn.Module):
    def __init__(self, rnn_cell, input_dim, embedding_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = rnn_cell(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # TODO: your codes here
        # text: [sent len, batch size]

        # 1. get embdding vectors
        # embedded: [sent len, batch size, emb dim]

        # 2. initialize hidden vector (considering special parts of LSTMCell)
        # hidden: [1, batch size, hid dim]

        # 3. multiple step recurrent forward

        # 4. get final output

        pass