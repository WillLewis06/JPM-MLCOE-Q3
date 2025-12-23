def generate_one_hot(probabilities):
    probabilities = np.array(probabilities)
    p_index = np.random.choice(len(probabilities), p=probabilities)
    one_hot = np.zeros_like(probabilities)
    one_hot[p_index] = 1
    return one_hot

product_set = [0, 1, 2, 3]
offer_set = [0, 1, 2, 3]
X = []
Y = []
np.random.seed(10)

hypothetical_choice_p = [[0.98, 0.02, 0, 0],
                         [0.5, 0, 0.5, 0],
                         [0.5, 0, 0, 0.5],
                         [0, 0.5, 0.5, 0],
                         [0, 0.5, 0, 0.5],
                         [0, 0, 0.9, 0.1],
                         [0.49, 0.01, 0.5, 0],
                         [0.49, 0.01, 0, 0.5],
                         [0.5, 0, 0.45, 0.05],
                         [0, 0.5, 0.45, 0.05],
                         [0.49, 0.01, 0.45, 0.05]]

index = 0
for r in range(2, len(offer_set) + 1):
    for subset in itertools.combinations(offer_set, r):
        binary_subset = [1 if x in subset else 0 for x in offer_set]
        p = hypothetical_choice_p[index]
        for _ in range(200):
            X.append(binary_subset)
            Y.append(generate_one_hot(p).reshape((1, len(product_set))))
        index += 1
        
import pandas as pd, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import os, math, random, json, datetime
from torch.utils.data import Subset
import torch, torch.nn as nn, torch.nn.functional as F, math
import functools


def make_valid_mask(n, lengths, device, mode='without'):
    row = torch.arange(n, device=device).unsqueeze(0)
    if mode == 'with':
        return (row < lengths.unsqueeze(1)) | (row == n - 1)   # (B,n) bool
    else:
        return  row < lengths.unsqueeze(1)

def masked_softmax(scores, lengths):        # scores:(B,n)
    n = scores.size(1)
    valid = make_valid_mask(n, lengths, scores.device)
    scores = scores.masked_fill(~valid, float('-inf'))
    return F.log_softmax(scores, dim=-1)


class NonlinearTransformation(nn.Module):
    def __init__(self, H, embed=128, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(embed, embed * H)
        self.fc2 = nn.Linear(embed, embed)
        self.H = H
        self.embed = embed
        self.dropout = nn.Dropout(dropout)
        self.enc_norm = nn.LayerNorm(embed)

    def forward(self, X):
        B, n, _ = X.shape
        X = self.fc1(X).view(B, n, self.H, self.embed)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.enc_norm(X)
        return X


class DeepHalo(nn.Module):
    def __init__(self, n, input_dim, H, L, embed=128, dropout=0):
        super().__init__()
        self.basic_encoder = nn.Sequential(
            nn.Linear(input_dim, embed), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed, embed),     nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed, embed)
        )
        self.enc_norm = nn.LayerNorm(embed)
        self.aggregate_linear  = nn.ModuleList([nn.Linear(embed, H) for _ in range(L)])
        self.nonlinear = nn.ModuleList([NonlinearTransformation(H, embed) for _ in range(L)])
        self.H = H
        self.embed = embed
        self.final_linear = nn.Linear(embed, 1)
        self.qualinear1 = nn.Linear(embed, embed)
        self.qualinear2 = nn.Linear(embed, embed)

    def forward(self, X, lengths):
        B, n, _ = X.shape
        Z = self.enc_norm(self.basic_encoder(X))
        X = Z.clone()
        for fc, nt in zip(self.aggregate_linear, self.nonlinear):
            Z_bar = (fc(Z).sum(1) / lengths.unsqueeze(1)).unsqueeze(-1).unsqueeze(1) #(B, 1, H, 1)
            phi = nt(X)
            valid = make_valid_mask(n, lengths, X.device)     # (B,n)
            # print(valid.shape, phi.shape)
            phi =  phi * valid.unsqueeze(-1).unsqueeze(-1)                                          #(B, n, H, embed)
            Z = (phi * Z_bar).sum(2) / self.H + Z

        logits = self.final_linear(Z).squeeze(-1)        # (B,n)
        probs  = masked_softmax(logits, lengths)          # log-probs
        return probs, logits



