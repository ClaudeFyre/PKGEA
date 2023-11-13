import torch.functional as F
from model import MGNN
import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import argparse
import os
import random
import pandas as pd

parser = argparse.ArgumentParser(description='hyperparameters')
parser.add_argument('--gpu',
                    default = '0',
                    type = str,
                    help='choose gpu device')
parser.add_argument('--file',
                    default='data/cn15k')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Load kgs
kg1_path = os.path.join(args.file, "KG1.csv")
kg2_path = os.path.join(args.file, "KG2.csv")
kg1 = pd.read_csv(kg1_path)
kg2 = pd.read_csv(kg2_path)

# Load Entity pairs
pair_path = os.path.join(args.file, "entity_pair.csv")
entity_pairs = pd.read_csv(pair_path)

# Shuffle the entity pairs
random.shuffle(entity_pairs)

# Calculate the sizes of train, val, and test sets
total_size = len(entity_pairs)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Split the entity pairs
train_pairs = entity_pairs[:train_size]
val_pairs = entity_pairs[train_size:train_size + val_size]
test_pairs = entity_pairs[train_size + val_size:]



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = dataset[0]
    model = MGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Get the embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.conv1(data.x, data.edge_index).numpy()

    # Compute transition probabilities from embeddings (using cosine similarity as an example)
    similarity_matrix = np.inner(embeddings, embeddings) / (
                np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(embeddings, axis=1))
    transition_probabilities = similarity_matrix / np.sum(similarity_matrix, axis=1)[:, None]
    # Now, you can use the transition_probabilities matrix to simulate a Markov Chain, as done in the previous example

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
