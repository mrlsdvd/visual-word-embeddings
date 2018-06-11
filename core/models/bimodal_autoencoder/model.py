import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pandas as pd
import csv
import sys, os
# Add path to config
sys.path.append('../../')
import config as conf


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return features


class Autoencoder(nn.Module):
    def __init__(self, embed_size, embeddings_path, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Autoencoder, self).__init__()
        # Load the trained model parameters
        embeddings = pd.read_table(embeddings_path, sep="\s+", index_col=0, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8').values
        # Add special token vecotors to embedding matrix
        print(embeddings.shape)
        special_embeddings = np.random.rand(4, embeddings.shape[1])
        embeddings = np.vstack([embeddings, special_embeddings])
        print(embeddings.shape)
        # Convert embedding to tensor
        embeddings = torch.tensor(embeddings).float()
        print(embeddings.shape)
        # Use pretrained embeddings
        self.embed = nn.Embedding(vocab_size, embed_size).from_pretrained(embeddings, freeze=True)
        self.linear_1 = nn.Linear(embed_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, embed_size)
        self.linear_3 = nn.Linear(embed_size, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, embed_size)
        self.tanh = nn.Tanh()

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # features: (N, D)
        # captions: (N, t)
        embeddings = self.embed(captions).detach() # embeddings: (N, t, D)
        N, T, D = embeddings.size()
        # packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True) # packed_embeddings: (N, T, D)
        # print(embeddings.shape)
        # print(packed_embeddings.shape)
        # print(features.shape)
        features = features.view(N, 1, D)
        features = features.repeat(1, T, 1) # packed_features: (N, T, D)
        # print(features.shape)
        h1 = self.tanh(self.linear_1(embeddings)) # h1: (N, T, H)
        h2 = self.tanh(self.linear_2(h1)) # h2: (N, T, D)
        h3 = self.tanh(self.linear_3(h2)) # h3: (N, T, H)
        h4 = self.tanh(self.linear_4(h3)) # h4: (N, T, D)

        # Compute L_ling and L_vis
        L_ling = torch.pow(embeddings - h4, 2)
        # print(L_ling.shape)
        L_ling = torch.sum(L_ling, dim=2) # L_ling: (N, T, 1)
        # print(L_ling.shape)
        L_ling = torch.mean(L_ling)
        # print(L_ling.shape)

        L_vis = torch.pow(features - h2, 2)
        L_vis = torch.sum(L_vis, dim=2)
        L_vis = torch.mean(L_vis)

        return L_ling, L_vis

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
